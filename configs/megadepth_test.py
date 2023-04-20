from src.config.default import _CN as cfg

TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "data/megadepth/test"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/megadepth_test_1500.txt"
cfg.DATASET.MGDPT_IMG_RESIZE = 1200
cfg.DATASET.MGDPT_IMG_PAD = True
cfg.DATASET.MGDPT_DF = 16
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

cfg.MODEL.COARSE.N_SAMPLES = 0
cfg.MODEL.MATCH_COARSE.THR = 0.25
cfg.MODEL.LOSS.FINE_TYPE = 'sym_epi'
