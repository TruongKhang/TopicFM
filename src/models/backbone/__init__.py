from .fpn import FPN
from .convnext import ConvNeXt, LayerNorm
from .convnext import Block as ConvNeXtBlock

def build_backbone(config):
    if config["backbone_type"] == "fpn":
        return FPN(config['fpn'])
    elif config["backbone_type"] == "convnext":
        return ConvNeXt(**config["convnext"])
    else:
        raise NotImplementedError("backbone network is unavailable")
