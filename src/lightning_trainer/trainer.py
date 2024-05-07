
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.models import TopicFM
from src.models.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.loss import TopicFMLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_Trainer(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, epoch_start=0):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.model_cfg = lower_config(_config['model'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)
        self.epoch_start = epoch_start

        # Matcher: TopicFM
        self.matcher = TopicFM(config=_config['model'])
        self.loss = TopicFMLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir

        self.validation_step_outputs = {}
        
    def configure_optimizers(self):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        lr = self.config.TRAINER.TRUE_LR
        if self.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (self.global_step / warmup_step) * abs(self.config.TRAINER.TRUE_LR - base_lr)
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')
        optimizer = build_optimizer(self, self.config, lr)
        scheduler = build_scheduler(self.config, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
    def _trainval_inference(self, batch):
        for data_dict in batch:
            data_dict["epoch_idx"] = self.trainer.current_epoch + self.epoch_start

            with self.profiler.profile("Compute coarse supervision"):
                compute_supervision_coarse(data_dict, self.config)
            
            with self.profiler.profile("TopicFM"):
                self.matcher(data_dict)
            
            with self.profiler.profile("Compute fine supervision"):
                compute_supervision_fine(data_dict, self.config)
        
        with self.profiler.profile("Compute losses"):
            loss, _ = self.loss(batch)
        
        return loss
    
    def _compute_metrics(self, batch):
        list_metrics = []
        with self.profiler.profile("Copmute metrics"):
            for data_dict in batch:
                compute_symmetrical_epipolar_errors(data_dict)  # compute epi_errs for each match
                compute_pose_errors(data_dict, self.config)  # compute R_errs, t_errs, pose_errs for each pair

                rel_pair_names = list(zip(*data_dict['pair_names']))
                bs = data_dict['image0'].size(0)
                metrics = {
                    'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                    'epi_errs': [data_dict['epi_errs'][data_dict['m_bids'] == b].cpu().numpy() for b in range(bs)],
                    'R_errs': data_dict['R_errs'],
                    't_errs': data_dict['t_errs'],
                    'inliers': data_dict['inliers']}
                list_metrics.append(metrics)

        ret_dict = {k: [] for k in list_metrics[0].keys()}
        for metrics in list_metrics:
            for k, v in metrics.items():
                ret_dict[k] += v

        return {"metrics": ret_dict}, rel_pair_names
    
    def training_step(self, batch_data, batch_idx):
        if isinstance(batch_data, torch.Tensor):
            batch_data = [batch_data]

        combined_loss = 0
        for batch in batch_data:
            loss = self._trainval_inference(batch)
            combined_loss = combined_loss + loss
        
        combined_loss = combined_loss / len(batch_data)
        
        self.log("loss", combined_loss, on_step=True, on_epoch=True, prog_bar=True, 
                 logger=True, sync_dist=True, batch_size=len(batch_data))

        return {"loss": combined_loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._trainval_inference(batch)
        
        ret_dict, _ = self._compute_metrics(batch)

        if dataloader_idx not in self.validation_step_outputs:
            self.validation_step_outputs[dataloader_idx] = [ret_dict]
        else:
            self.validation_step_outputs[dataloader_idx].append(ret_dict)

        return {
            **ret_dict
        }

    def on_validation_epoch_end(self):
        multi_val_metrics = defaultdict(list)
        for idx, outputs in self.validation_step_outputs.items():
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList([_me[k] for _me in _metrics]) for k in _metrics[0]}
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'dataset{idx}_auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
        
        for idx in self.validation_step_outputs.keys():
            for thr in [5, 10, 20]:
                self.log(f'dataset{idx}_auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'dataset{idx}_auc@{thr}'])), sync_dist=True)
        
        self.validation_step_outputs.clear()
