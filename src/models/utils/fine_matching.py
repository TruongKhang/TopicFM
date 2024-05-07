import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

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

        feat_f0_picked = feat_f0[:, WW//2, :]

        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)
        feat_f1_picked = (feat_f1 * heatmap.unsqueeze(-1)).sum(dim=1) # [M, C]
        heatmap = heatmap.view(-1, W, W)

        # compute coordinates from heatmap
        coords1_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords1_normalized**2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability
        
        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords1_normalized, std.unsqueeze(1)], -1),
                     'descriptors0': feat_f0_picked.detach(), 'descriptors1': feat_f1_picked.detach()})

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


class DynamicFineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

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
        self.M, self.W, self.WW, self.C= M, W, WW, C
        self.scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        self.scale1 = scale * data['scale1'][data['b_ids']] if 'scale1' in data else scale
        # grid_WW = create_meshgrid(W, W, normalized_coordinates=True, device=feat_f0.device).reshape(1, -1, 2).repeat(M, 1, 1)

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'f_b_ids': data['b_ids'],
                'all_mkpts0_f': data['all_mkpts0_c'],
                'all_mkpts1_f': data['all_mkpts1_c'],
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        temperature = 0.1
        score_map0 = data["score_map0"]
        if self.training:
            temperature = math.pow(2, -0.5*data["epoch_idx"]) if data["epoch_idx"] < 8 else 0.1
        heatmap0 = F.softmax(score_map0 / temperature, dim=-1)
        feat_f0_picked = (feat_f0 * heatmap0.unsqueeze(-1)).sum(dim=1)
        coords0_normed = dsnt.spatial_expectation2d(heatmap0.view(-1, W, W)[None], True)[0]
        out = self.get_fine_match(data, coords0_normed, feat_f0_picked, feat_f1)
        out["mconf"] *= data["mconf"]
        data.update(out)

    def get_fine_match(self, data, coords0_normed, feat_f0_picked, feat_f1, mask_b_ids=None):
        if mask_b_ids is None:
            mask_b_ids = torch.ones_like(data["b_ids"]).bool()
        coords0_normed = coords0_normed[mask_b_ids]
        feat_f0_picked = feat_f0_picked[mask_b_ids]
        feat_f1 = feat_f1[mask_b_ids]

        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1) / self.C ** .5
        heatmap1 = torch.softmax(sim_matrix, dim=1)
        mconf, _ = torch.max(heatmap1, dim=1)
        # data.update({"heatmap1": heatmap1})
        # feat_f1_picked = (feat_f1 * heatmap1.unsqueeze(-1)).sum(dim=1)  # [M, C]
        heatmap1 = heatmap1.view(-1, self.W, self.W)

        # compute coordinates from heatmap
        coords1_normed = dsnt.spatial_expectation2d(heatmap1[None], True)[0]
        scale0 = self.scale0[mask_b_ids] if isinstance(self.scale0, torch.Tensor) else self.scale0
        all_mkpts0_f = data['all_mkpts0_c'][mask_b_ids] + (coords0_normed * (self.W // 2) * scale0)
        scale1 = self.scale1[mask_b_ids] if isinstance(self.scale1, torch.Tensor) else self.scale1
        all_mkpts1_f = data['all_mkpts1_c'][mask_b_ids] + (coords1_normed * (self.W // 2) * scale1)

        # mkpts0_f and mkpts1_f
        true_matches = (~data["gt_mask"])[mask_b_ids]
        mkpts0_f = all_mkpts0_f.detach()[true_matches]
        mkpts1_f = all_mkpts1_f.detach()[true_matches]
        # scale0_f, scale1_f = scale0[true_matches], scale1[true_matches]
        scale0_f = scale0[true_matches] if isinstance(scale0, torch.Tensor) else scale0
        scale1_f = scale1[true_matches] if isinstance(scale1, torch.Tensor) else scale1
        mconf = mconf.detach()[true_matches]
        f_b_ids = data["b_ids"][mask_b_ids]

        return {
            "f_b_ids": f_b_ids,
            "m_bids": f_b_ids[true_matches],
            "mconf": mconf,
            "all_mkpts0_f": all_mkpts0_f,
            "all_mkpts1_f": all_mkpts1_f,
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            "scale0_f": scale0_f, "scale1_f": scale1_f
        }
