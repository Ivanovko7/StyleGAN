import torch.nn as nn
import torch.nn.functional as F
from models.conv_blocks import InvertedResBlock
from models.conv_blocks import ConvBlock
from lib.common import initialize_weights


class StyleGen(nn.Module):
    def __init__(self, dataset=''):
        super(StyleGen, self).__init__()
        self.name = f'{self.__class__.__name__}_{dataset}'

        self.encoder = nn.Sequential(
            ConvBlock(3, 32, kernel_size=7, stride=1, padding=3, norm_type="layer"),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=(0, 1, 0, 1), norm_type="layer"),
            ConvBlock(64, 64, kernel_size=3, stride=1, norm_type="layer"),
        )

        self.downsample = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=(0, 1, 0, 1), norm_type="layer"),
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer"),
        )

        self.residual_blocks = nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer"),
            InvertedResBlock(128, 256, expand_ratio=2, norm_type="layer"),
            InvertedResBlock(256, 256, expand_ratio=2, norm_type="layer"),
            InvertedResBlock(256, 256, expand_ratio=2, norm_type="layer"),
            InvertedResBlock(256, 256, expand_ratio=2, norm_type="layer"),
            ConvBlock(256, 128, kernel_size=3, stride=1, norm_type="layer"),
        )

        self.convert_blocks = nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer"),
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer"),
        )

        self.upsample = nn.Sequential(
            ConvBlock(128, 64, kernel_size=3, stride=1, norm_type="layer"),
            ConvBlock(64, 64, kernel_size=3, stride=1, norm_type="layer"),
            ConvBlock(64, 32, kernel_size=7, padding=3, stride=1, norm_type="layer"),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, x):
        out = self.encoder(x)
        out = self.downsample(out)
        out = self.residual_blocks(out)
        out = F.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.convert_blocks(out)
        out = F.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.upsample(out)
        img = self.decoder(out)

        return img
