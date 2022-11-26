from src.config.default import _CN as cfg

cfg.MODEL.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.TRAINER.CANONICAL_LR = 1e-4
cfg.TRAINER.MSLR_MILESTONES = [4, 8, 12, 17, 20, 23, 26, 29]

cfg.MODEL.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2
