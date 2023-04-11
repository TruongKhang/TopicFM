import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from src.models.modules import LoFTREncoderLayer

INF = 1e9

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

        self.num_coarse_matches = config["num_coarse_matches"]
        self.geometric_verify = config["geo_match"]
        if config["geo_match"]:
            # geometric verification
            self.rel_pos_enc = nn.Sequential(nn.Linear(2, 256, bias=True),
                                             nn.GELU(),
                                             nn.Linear(256, 256, bias=False),
                                             nn.LayerNorm(256))
            self.n_attn_iters = 4
            self.attn_enc = nn.ModuleList([LoFTREncoderLayer(256, 8, attention='linear')
                                           for _ in range(self.n_attn_iters * 2)])
            pos_enc = positionalencoding1d(256, 5000)
            self.register_buffer("pos_enc", pos_enc)

    def geo_match(self, b_ids, i_ids, j_ids, labels, data, num_pairs=None):
        device, dtype = b_ids.device, data["image0"].dtype
        labels = labels.to(dtype)
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode="floor")], dim=1).to(dtype)
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode="floor")], dim=1).to(dtype)
        if num_pairs is not None:
            if (not self.training) and len(b_ids) > 0:
                _, count_matches = torch.unique(b_ids, return_counts=True)
                num_pairs = torch.max(count_matches)
            coords0 = torch.zeros((data["bs"], num_pairs, 2), device=device, dtype=dtype)
            coords1 = torch.zeros_like(coords0)
            mask0, mask1 = torch.zeros_like(coords0[..., 0]), torch.zeros_like(coords1[..., 0])
            pe = torch.zeros((data["bs"], num_pairs, self.pos_enc.shape[-1]), device=device, dtype=dtype)
            m_labels = torch.zeros((data["bs"], num_pairs), device=device, dtype=dtype)

            for bidx in range(data["bs"]):
                mask_b_ids = (b_ids == bidx)
                mkpts0_bidx, mkpts1_bidx = mkpts0_c[mask_b_ids], mkpts1_c[mask_b_ids]
                pe_bidx = self.pos_enc[:len(mkpts0_bidx)]
                labels_bidx = labels[mask_b_ids]

                if len(mkpts0_bidx) > num_pairs:
                    if self.training:
                        random_ids = torch.multinomial(1 - labels_bidx+0.1, num_pairs, replacement=False)
                        mkpts0_bidx, mkpts1_bidx = mkpts0_bidx[random_ids], mkpts1_bidx[random_ids]
                        pe_bidx = pe_bidx[random_ids]
                        labels_bidx = labels_bidx[random_ids]
                    else:
                        mkpts0_bidx, mkpts1_bidx = mkpts0_bidx[:num_pairs], mkpts1_bidx[:num_pairs]
                        pe_idx = pe_bidx[:num_pairs]
                        labels_bidx = labels_bidx[:num_pairs]

                coords0[bidx][:len(mkpts0_bidx)] += mkpts0_bidx
                mask0[bidx][:len(mkpts0_bidx)] = 1.0
                coords1[bidx][:len(mkpts1_bidx)] += mkpts1_bidx
                mask1[bidx][:len(mkpts1_bidx)] = 1.0
                pe[bidx][:len(pe_bidx)] += pe_bidx
                m_labels[bidx][:len(labels_bidx)] += labels_bidx
        else:
            coords0, coords1 = mkpts0_c.unsqueeze(0).to(dtype), mkpts1_c.unsqueeze(0).to(dtype)
            mask0, mask1 = torch.ones_like(coords0[:, :, 0]), torch.ones_like(coords1[:, :, 0])
            pe = self.pos_enc[:len(mkpts0_c)].unsqueeze(0)
            m_labels = labels.unsqueeze(0)

        # normalize coordinate
        scale0_c2o = data["scale0"] * data['hw0_i'][0] / data['hw0_c'][0]
        scale1_c2o = data["scale1"] * data['hw1_i'][0] / data['hw1_c'][0]
        norm_coords0 = self.norm_coords(coords0, data["K0"], scale0_c2o)
        norm_coords1 = self.norm_coords(coords1, data["K1"], scale1_c2o)
        # if self.training:
        #     print(len(b_ids), coords0.shape, coords1.shape)
        #     print("coords0: ", coords0[0, :10].cpu().numpy(), norm_coords0[0, :10].cpu().numpy(), mask0[0, :10].cpu().numpy())
        #     print("coords1: ", coords1[0, :10].cpu().numpy(), norm_coords1[0, :10].cpu().numpy(), mask1[0, :10].cpu().numpy())
        geo_feat0 = self.rel_pos_enc(norm_coords0) + pe
        geo_feat1 = self.rel_pos_enc(norm_coords1) + pe
        # geo_feat0[:, :100] = geo_feat0[:, :100] + pe[:, :100]
        # geo_feat1[:, :100] = geo_feat1[:, :100] + pe[:, :100]
        for idx in range(self.n_attn_iters):
            geo_feat0 = self.attn_enc[idx*2](geo_feat0, geo_feat0, mask0, mask0)
            geo_feat1 = self.attn_enc[idx*2](geo_feat1, geo_feat1, mask1, mask1)
            geo_feat0 = self.attn_enc[idx*2+1](geo_feat0, geo_feat1, mask0, mask1)
            geo_feat1 = self.attn_enc[idx*2+1](geo_feat1, geo_feat0, mask1, mask0)

        """geo_match_matrix = torch.einsum("bmd,bnd->bmn", geo_feat0 / 16, geo_feat1 / 16) * 16
        valid_geo_mask = (mask0[..., None] * mask1[:, None]).bool()
        geo_match_matrix.masked_fill_(~valid_geo_mask, -1e9)
        geo_conf_matrix = F.softmax(geo_match_matrix, dim=1) * F.softmax(geo_match_matrix, dim=2)
        geo_conf_pairs = geo_conf_matrix[:, range(coords0.shape[1]), range(coords0.shape[1])]"""
        geo_conf_pairs = (geo_feat0 * geo_feat1).sum(dim=-1) / 16
        valid_pairs = mask0.bool()

        # return geo_conf_matrix, geo_conf_pairs, valid_pairs, coords0, coords1
        return geo_conf_pairs, m_labels, valid_pairs, coords0, coords1

    def norm_coords(self, coords, K, scale):
        coords = coords * scale.unsqueeze(1)
        homo_coords = torch.cat((coords, torch.ones_like(coords[:, :, [0]])), dim=-1)
        homo_coords = homo_coords.permute(0, 2, 1)
        normed_coords = torch.inverse(K) @ homo_coords
        normed_coords = normed_coords.permute(0, 2, 1)
        normed_coords -= (normed_coords[:, [0]] - 0.01)
        return normed_coords[:, :, :2]

    def forward(self, data):
        """
        Args:
            data (dict)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        conf_matrix = data['conf_matrix']
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        conf_mask = conf_matrix > self.thr
        conf_mask = rearrange(conf_mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(conf_mask, self.border_rm, False)
        else:
            mask_border_with_padding(conf_mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        conf_mask = rearrange(conf_mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # 2. mutual nearest

        conf_mask = conf_mask \
            * torch.zeros_like(conf_matrix).scatter_(1, conf_matrix.topk(k=2, dim=1)[1], 1.0) \
            * torch.zeros_like(conf_matrix).scatter_(2, conf_matrix.topk(k=2, dim=2)[1], 1.0)
        # b_ids, i_ids, j_ids = torch.nonzero(mask, as_tuple=True)

        if self.training and self.geometric_verify:
            gt_bids, gt_iids, gt_jids = data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids']
            non_m_bids, non_m_iids, non_m_jids = torch.nonzero((conf_mask.int() - data["conf_matrix_gt"].int()) == 1, as_tuple=True)
            b_ids, i_ids, j_ids = map(lambda x, y: torch.cat((x, y), dim=0),
                                      *zip([non_m_bids, gt_bids], [non_m_iids, gt_iids], [non_m_jids, gt_jids]))
            labels = torch.ones_like(b_ids)
            labels[:len(non_m_bids)] = 0.0
            geo_conf_pairs, labels, valid_pairs, _, _ = self.geo_match(b_ids, i_ids, j_ids, labels, data,
                                                                  num_pairs=self.num_coarse_matches)

            data.update({"geo_conf_pairs": geo_conf_pairs, "geo_labels": labels, "valid_pairs": valid_pairs})
        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_mask, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_mask, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_mask.device
        # 1. confidence thresholding
        """conf_mask = conf_matrix > self.thr
        conf_mask = rearrange(conf_mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(conf_mask, self.border_rm, False)
        else:
            mask_border_with_padding(conf_mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        conf_mask = rearrange(conf_mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # 2. mutual nearest

        mask = mask \
            * torch.zeros_like(conf_matrix).scatter_(1, conf_matrix.topk(k=2, dim=1)[1], 1.0) \
            * torch.zeros_like(conf_matrix).scatter_(2, conf_matrix.topk(k=2, dim=2)[1], 1.0)"""
        b_ids, i_ids, j_ids = torch.nonzero(conf_mask, as_tuple=True)
        mconf = data["conf_matrix"][b_ids, i_ids, j_ids]

        mconf, sorted_ids = torch.sort(mconf, dim=0, descending=True)

        # if len(mconf) > self.num_coarse_matches:
        #     mconf, sorted_ids = mconf[:self.num_coarse_matches], sorted_ids[:self.num_coarse_matches] # torch.topk(mconf, self.num_coarse_matches)
        b_ids, i_ids, j_ids = b_ids[sorted_ids], i_ids[sorted_ids], j_ids[sorted_ids]

        if self.geometric_verify:
            # geometric refinement
            geo_conf_pairs, _, valid_pairs, m_coords0, m_coords1 = self.geo_match(b_ids, i_ids, j_ids, torch.ones_like(b_ids), data,
                                                                               num_pairs=self.num_coarse_matches)
            # mutual_nearest = (geo_conf_mat == geo_conf_mat.max(dim=2, keepdim=True)[0]) \
            #                * (geo_conf_mat == geo_conf_mat.max(dim=1, keepdim=True)[0])
            # geo_conf_pairs = mutual_nearest[:, range(m_coords0.shape[1]), range(m_coords0.shape[1])]
            geo_mask = (torch.sigmoid(geo_conf_pairs) > 0.5) & valid_pairs
            b_ids, _ = torch.nonzero(geo_mask, as_tuple=True)
            mconf = geo_conf_pairs[geo_mask] # .float()
            out_coords0, out_coords1 = m_coords0[geo_mask], m_coords1[geo_mask]
            i_ids = (out_coords0[:, 0] + out_coords0[:, 1] * data["hw0_c"][1]).long()
            j_ids = (out_coords1[:, 0] + out_coords1[:, 1] * data["hw1_c"][1]).long()

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = conf_mask.size(0) * max(conf_mask.size(1), conf_mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min, ),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                    len(data['spv_b_ids']),
                    (max(num_matches_train - num_matches_pred,
                        self.train_pad_num_gt_min), ),
                    device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode="floor")],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode="floor")],
            dim=1) * scale1

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'all_mkpts0_c': mkpts0_c,
            'all_mkpts1_c': mkpts1_c,
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches
