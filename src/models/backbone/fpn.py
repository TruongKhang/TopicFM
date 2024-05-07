import torch.nn as nn
import torch.nn.functional as F

from .convnext import LayerNorm

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, padding=1, stride=1, norm="bn"):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(planes, eps=1e-6)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(planes, affine=True)
        elif norm == "ln":
            self.norm = LayerNorm(planes, data_format="channels_first")
        else:
            self.norm = None

        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv(x)
        if self.norm:
            y = self.norm(y)
        y = self.act(y)
        return y


class FPN(nn.Module):
    """
    FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = ConvBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        self.layer0 = ConvBlock(3, initial_dim, kernel_size=7, padding=3, stride=2)
        self.layer1 = self._make_layer(block, initial_dim, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[0], block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[1], block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[2], block_dims[3], stride=2)  # 1/16

        self.layer3_outconv = nn.Conv2d(block_dims[2], block_dims[3], kernel_size=1, padding=0, bias=False)
        self.layer3_outconv2 = nn.Sequential(
            ConvBlock(block_dims[3], block_dims[2]),
            # ConvBlock(block_dims[2], block_dims[2]),
            nn.Conv2d(block_dims[2], block_dims[2], kernel_size=3, padding=1, bias=False),
        )
        self.norm_outlayer3 = LayerNorm(block_dims[2], eps=1e-6, data_format="channels_first")

        self.layer2_outconv = nn.Conv2d(block_dims[1], block_dims[2], kernel_size=1, padding=0, bias=False)
        self.layer2_outconv2 = nn.Sequential(
            ConvBlock(block_dims[2], block_dims[1]),
            nn.Conv2d(block_dims[1], block_dims[1], kernel_size=3, padding=1, bias=False),
        )
        self.layer1_outconv = nn.Conv2d(block_dims[0], block_dims[1], kernel_size=1, padding=0, bias=False)
        self.layer1_outconv2 = nn.Sequential(
            ConvBlock(block_dims[1], block_dims[0]),
            nn.Conv2d(block_dims[0], block_dims[0], kernel_size=3, padding=1, bias=False),
        )
        self.norm_outlayer1 = LayerNorm(block_dims[0], eps=1e-6, data_format="channels_first")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_dim, out_dim, kernel_size=3, padding=1, stride=1):
        layer1 = block(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        layer2 = block(out_dim, out_dim, stride=1)
        layers = (layer1, layer2)
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.layer0(x) #self.act(self.bn1(self.conv1(x))))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN
        x4_out_2x = F.interpolate(x4, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        return [self.norm_outlayer3(x3_out), self.norm_outlayer1(x1_out)]
