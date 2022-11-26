import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self): #, ww, c):
        super().__init__()
        """self.fc = nn.Sequential(nn.Linear(ww*c, ww*c//4, bias=False),
                                # nn.LayerNorm(ww*c//8),
                                nn.GELU(),
                                nn.Linear(ww*c//4, ww*c//16, bias=False),
                                nn.GELU(),
                                nn.Linear(ww*c//16, ww, bias=False))"""

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        """score_map0 = self.fc(feat_f0.reshape(M, -1))
        if self.training:
            prob_map0 = F.gumbel_softmax(score_map0, tau=0.1, hard=True)
            # feat_f0_picked = (feat_f0 * prob_map0.unsqueeze(-1)).sum(dim=1) #feat_f0[:, WW//2, :]
        else:
            prob_map0 = F.softmax(score_map0 / 0.1, dim=-1)"""
            # selected_ids = torch.argmax(score_map0, dim=-1, keepdim=True)
            # feat_f0_picked = torch.gather(feat_f0, dim=1, index=selected_ids.unsqueeze(-1).repeat(1, 1, C))
            # feat_f0_picked = feat_f0_picked.squeeze(1)
        feat_f0_picked = feat_f0[:, WW//2, :] #(feat_f0 * prob_map0.unsqueeze(-1)).sum(dim=1)
        #coords0_normed = dsnt.spatial_expectation2d(prob_map0.detach().view(1, -1, W, W), True)[0]

        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords1_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords1_normalized**2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability
        
        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords1_normalized, std.unsqueeze(1)], -1)})

        # compute absolute kpt coords
        self.get_fine_match(coords1_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords1_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        # scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        mkpts0_f = data['mkpts0_c'] # + (coords0_normed * (W // 2) * scale0 )[:len(data['mconf'])]
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale1' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords1_normed * (W // 2) * scale1)[:len(data['mconf'])]

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })
