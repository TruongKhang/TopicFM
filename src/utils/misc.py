import os
import numpy as np
from typing import Union
from loguru import _Logger, logger
from itertools import chain

import torch
from yacs.config import CfgNode as CN
from pytorch_lightning.utilities import rank_zero_only


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def log_on(condition, message, level):
    if condition:
        assert level in ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']
        logger.log(level, message)


def get_rank_zero_only_logger(logger: _Logger):
    if rank_zero_only.rank == 0:
        return logger
    else:
        for _level in logger._core.levels.keys():
            level = _level.lower()
            setattr(logger, level,
                    lambda x: None)
        logger._log = lambda x: None
    return logger


def setup_gpus(gpus: Union[str, int]) -> int:
    """ A temporary fix for pytorch-lighting 1.3.x """
    gpus = str(gpus)
    gpu_ids = []
    
    if ',' not in gpus:
        n_gpus = int(gpus)
        return n_gpus if n_gpus != -1 else torch.cuda.device_count()
    else:
        gpu_ids = [i.strip() for i in gpus.split(',') if i != '']
    
    # setup environment variables
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_devices is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in gpu_ids)
        visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        logger.warning(f'[Temporary Fix] manually set CUDA_VISIBLE_DEVICES when specifying gpus to use: {visible_devices}')
    else:
        logger.warning('[Temporary Fix] CUDA_VISIBLE_DEVICES already set by user or the main process.')
    return len(gpu_ids)


def flattenList(x):
    return list(chain(*x))


def make_recursive_func(func):
    def wrapper(vars, dtype):
        if isinstance(vars, list):
            return [wrapper(x, dtype) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, dtype) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, dtype) for k, v in vars.items()}
        else:
            return func(vars, dtype)

    return wrapper


@make_recursive_func
def totensor(vars, dtype):
    if isinstance(vars, torch.Tensor):
        return vars.unsqueeze(0).to(dtype)
    elif isinstance(vars, np.ndarray):
        return torch.from_numpy(vars).unsqueeze(0).to(dtype)
    elif isinstance(vars, str):
        return [vars]
    elif isinstance(vars, (int, float)):
        return torch.tensor([vars])
    else:
        raise NotImplementedError("invalid input type {}".format(type(vars)))
