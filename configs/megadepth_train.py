from src.config.default import _CN as cfg

TRAIN_BASE_PATH = "data/megadepth/index"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TRAIN_DATA_ROOT = "data/megadepth/train"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/trainvaltest_list/train_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.01

TEST_BASE_PATH = "data/megadepth/index"
cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/megadepth/test"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_val_1500"
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/trainvaltest_list/val_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0   # for both test and val
cfg.DATASET.MGDPT_IMG_RESIZE = 800 # for training on 11GB mem GPUs
cfg.DATASET.MGDPT_IMG_PAD = True
cfg.DATASET.MGDPT_DF = 16

# cfg.MODEL.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.MODEL.COARSE.N_SAMPLES = 0
cfg.MODEL.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
# cfg.MODEL.MATCH_COARSE.NUM_COARSE_MATCHES = None
cfg.MODEL.LOSS.FINE_TYPE = 'sym_epi'

cfg.TRAINER.CANONICAL_LR = 1e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [4, 8, 12, 16, 20, 25, 30]
# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5
cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
# 368 scenes in total for MegaDepth
# (with difficulty balanced (further split each scene to 3 sub-scenes))
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 50
