from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.epipolar import sampson_epipolar_distance, symmetrical_epipolar_distance


def sample_non_matches(pos_mask, match_ids=None, sampling_ratio=10):
    # assert (pos_mask.shape == mask.shape) # [B, H*W, H*W]
    if match_ids is not None:
        HW = pos_mask.shape[1]
        b_ids, i_ids, j_ids = match_ids
        if len(b_ids) == 0:
            return ~pos_mask

        neg_mask = torch.zeros_like(pos_mask)
        probs = torch.ones((HW - 1)//3, device=pos_mask.device)
        for _ in range(sampling_ratio):
            d = torch.multinomial(probs, len(j_ids), replacement=True)
            sampled_j_ids = (j_ids + d*3 + 1) % HW
            neg_mask[b_ids, i_ids, sampled_j_ids] = True
        # neg_mask = neg_matrix == 1
    else:
        neg_mask = ~pos_mask

    return neg_mask


class TopicFMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['model']['loss']
        # self.match_type = self.config['model']['match_coarse']['match_type']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def compute_coarse_loss(self, data, match_ids=None, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        conf, conf_gt = data['conf_matrix'], data['conf_matrix_gt']
        topic_mat = data['topic_matrix']

        pos_mask = conf_gt == 1
        neg_mask = sample_non_matches(pos_mask, match_ids=match_ids)
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.loss_config['focal_alpha']

        loss = 0.0
        if isinstance(topic_mat, torch.Tensor):
            pos_topic = topic_mat[pos_mask]
            loss_pos_topic = - alpha * (pos_topic + 1e-6).log()
            neg_topic = topic_mat[neg_mask]
            loss_neg_topic = - alpha * (1 - neg_topic + 1e-6).log()
            if weight is not None:
                loss_pos_topic = loss_pos_topic * weight[pos_mask]
                loss_neg_topic = loss_neg_topic * weight[neg_mask]
            loss = loss_pos_topic.mean() + loss_neg_topic.mean()

        pos_conf = conf[pos_mask]
        loss_pos = - alpha * pos_conf.log()
        # handle loss weights
        if weight is not None:
            # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
            # but only through manually setting corresponding regions in sim_matrix to '-inf'.
            loss_pos = loss_pos * weight[pos_mask]

        loss = loss + c_pos_w * loss_pos.mean()

        if "geo_conf_pairs" in data:
            # geometric loss
            geo_conf_pairs, geo_labels, valid_pairs = data["geo_conf_pairs"], data["geo_labels"], data["valid_pairs"]
            # loss_geo = - alpha * torch.log(geo_conf_pairs[valid_pairs] + 1e-5)
            num_pos = geo_labels[valid_pairs].sum()
            num_neg = valid_pairs.sum() - num_pos
            pos_weight = float(num_neg / num_pos) if num_neg > 0 else 1.0
            data["num_neg_matches"] = num_neg
            pos_weight = torch.tensor([float(num_neg/num_pos)], device=geo_labels.device)
            loss_geo = F.binary_cross_entropy_with_logits(geo_conf_pairs[valid_pairs], geo_labels[valid_pairs], pos_weight=pos_weight)
            loss = loss + loss_geo # .mean()

        return loss
        
    def compute_fine_loss(self, **kwargs):
        if self.fine_type == 'l2_with_std':
            expec_f, expec_f_gt = kwargs["expec_f"], kwargs["expec_f_gt"]
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            expec_f, expec_f_gt = kwargs["expec_f"], kwargs["expec_f_gt"]
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        elif self.fine_type == 'sym_epi':
            f_kpts0, f_kpts1 = kwargs["f_kpts0"], kwargs["f_kpts1"]
            FMat, heatmap0 = kwargs["FMat"], kwargs["heatmap0"]
            return self._compute_sym_epipolar_distance(f_kpts0, f_kpts1, FMat, heatmap0)
        elif self.fine_type == 'sampson':
            f_kpts0, f_kpts1 = kwargs["f_kpts0"], kwargs["f_kpts1"]
            FMat = kwargs["FMat"]
            return self._compute_sampson_distance(f_kpts0, f_kpts1, FMat)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss

    def _compute_sym_epipolar_distance(self, kpts0, kpts1, FMat, heatmap=None, dist_thr=100):
        sym_dist = symmetrical_epipolar_distance(kpts0.unsqueeze(1), kpts1.unsqueeze(1), FMat, squared=False,
                                                 eps=1e-6).squeeze(-1)
        """mask = sym_dist.detach() < dist_thr
        entropy = (heatmap * torch.log(heatmap + 1e-6)).sum(dim=-1) if heatmap is not None else torch.zeros_like(sym_dist)
        inlier_loss = torch.mean(sym_dist[mask] - 0.2 * entropy[mask])
        outlier_loss = torch.mean(entropy[~mask])
        loss = inlier_loss + outlier_loss"""
        loss = sym_dist.clamp(min=0, max=100) * 0.25
        return loss.mean()

    def _compute_sampson_distance(self, kpts0, kpts1, FMat):
        loss = sampson_epipolar_distance(kpts0.unsqueeze(1), kpts1.unsqueeze(1), FMat, squared=True, eps=1e-6)
        loss = loss.clamp(min=0, max=100) * 0.25
        return loss.mean()
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(data, match_ids=(data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids']),
                                          weight=c_weight)
        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        if self.fine_type in ["l2", "l2_with_std"]:
            loss_f = self.compute_fine_loss(expec_f=data['expec_f'], expec_f_gt=data['expec_f_gt'])
        else:
            loss_f = self.compute_fine_loss(f_kpts0=data["all_mkpts0_f"], f_kpts1=data["all_mkpts1_f"],
                                            FMat=data["FMat_f"])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
