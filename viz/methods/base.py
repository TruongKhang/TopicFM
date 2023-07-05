import pprint
from abc import ABCMeta, abstractmethod
import torch
from itertools import chain
import cv2

from src.utils.plotting import make_matching_figure, error_colormap
from src.utils.metrics import aggregate_metrics


def flatten_list(x):
    return list(chain(*x))


class Viz(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)

        # for evaluation metrics of MegaDepth and ScanNet
        self.eval_stats = []
        self.time_stats = []
        self.flops_stats = {"backbone": [], "coarse_net": [], "fine_net": [], "total": []}

    def draw_matches(self, mkpts0, mkpts1, img0, img1, conf, path=None, **kwargs):
        thr = 5e-4
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        if "conf_thr" in kwargs:
            thr = kwargs["conf_thr"]
        color = error_colormap(conf, thr, alpha=0.1)

        text = [
            f"{self.name}",
            f"#Matches: {len(mkpts0)}",
        ]
        if 'R_errs' in kwargs:
            text.append(f"$\\Delta$R:{kwargs['R_errs']:.2f}°,  $\\Delta$t:{kwargs['t_errs']:.2f}°",)

        if path:
            make_matching_figure(img0, img1, mkpts0, mkpts1, color, text=text, path=path, dpi=150)
        else:
            return make_matching_figure(img0, img1, mkpts0, mkpts1, color, text=text)

    @abstractmethod
    def match_and_draw(self, data_dict, **kwargs):
        pass

    def compute_eval_metrics(self, epi_err_thr=5e-4):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in self.eval_stats]
        metrics = {k: flatten_list([_me[k] for _me in _metrics]) for k in _metrics[0]}

        val_metrics_4tb = aggregate_metrics(metrics, epi_err_thr)
        print('\n' + pprint.pformat(val_metrics_4tb))

    def measure_time(self):
        if len(self.time_stats) == 0:
            return 0
        return sum(self.time_stats) / len(self.time_stats)

    def measure_flops(self):
        outputs = {}
        for k, v in self.flops_stats.items():
            if len(v) == 0:
                outputs[k] = 0
            else:
                outputs[k] = sum(v) / len(v)
        return outputs
