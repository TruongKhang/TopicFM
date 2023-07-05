import torch
import torch.nn as nn
from einops.einops import rearrange
# from fvcore.nn import FlopCountAnalysis

from .backbone import build_backbone
from .modules import FineNetwork, FinePreprocess, TopicFormer
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching, DynamicFineMatching


class TopicFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)

        self.coarse_net = TopicFormer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)

        if config["loss"]["fine_type"] in ["l2_with_std", "l2"]:
            self.fine_net = FineNetwork(config["fine"], add_detector=False)
            self.fine_matching = FineMatching()
        else:
            self.fine_net = FineNetwork(config["fine"], add_detector=True)
            self.fine_matching = DynamicFineMatching()

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
            # backbone_counter = FlopCountAnalysis(self.backbone, torch.cat([data['image0'], data['image1']], dim=0))
            # backbone_flops = backbone_counter.total() / 1e9
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])
            # backbone_counter0 = FlopCountAnalysis(self.backbone, data['image0'])
            # backbone_counter1 = FlopCountAnalysis(self.backbone, data['image1'])
            # backbone_flops = backbone_counter0.total() / 1e9 + backbone_counter1.total() / 1e9

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:], 
            # "backbone_flops": backbone_flops,
        })

        # 2. coarse-level loftr module
        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        # coarse_net_counter = FlopCountAnalysis(self.coarse_net, (feat_c0, feat_c1))
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
            # coarse_net_counter = FlopCountAnalysis(self.coarse_net, (feat_c0, feat_c1, mask_c0, mask_c1))

        feat_c0, feat_c1, topic_matrix = self.coarse_net(feat_c0, feat_c1, mask_c0, mask_c1)
        data.update({"topic_matrix": topic_matrix,
                     # "coarse_net_flops": coarse_net_counter.total() / 1e9,
                     })

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, mask_c0, mask_c1, data)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0.detach(), feat_c1.detach(), data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold, score_map0 = self.fine_net(feat_f0_unfold, feat_f1_unfold)
            # fine_net_counter = FlopCountAnalysis(self.fine_net, (feat_f0_unfold, feat_f1_unfold))
            if score_map0 is not None:
                data["score_map0"] = score_map0
        # data.update({"fine_net_flops": fine_net_counter.total() / 1e9})

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
