
import torch.nn as nn
from models.conv_blocks import InvertedResBlock
from models.conv_blocks import ConvBlock
from models.conv_blocks import UpConvLNormLReLU
from lib.common import initialize_weights


class StyleGen(nn.Module):
    def __init__(self, dataset_name: str = ''):
        """
        Initialize the StyleGen model with the given dataset name.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset used to train the model.

        Notes
        -----
        The model is initialized with the `initialize_weights` function, which
        sets the weights of the model to random values.
        """
        super().__init__()
        self.name = f"{self.__class__.__name__}_{dataset_name}"
        self.bias = False

        self.encoder = nn.Sequential(
            ConvBlock(3, 32, kernel_size=7, stride=1, norm_type="layer", bias=self.bias),
            ConvBlock(32, 64, kernel_size=3, stride=2, norm_type="layer", bias=self.bias),
            ConvBlock(64, 64, kernel_size=3, stride=1, norm_type="layer", bias=self.bias),
        )

        self.downsample = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, norm_type="layer", bias=self.bias),
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer", bias=self.bias),
        )

        self.residual_blocks = nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer", bias=self.bias),
            InvertedResBlock(128, 256, expand_ratio=2, norm_type="layer", bias=self.bias),
            InvertedResBlock(256, 256, expand_ratio=2, norm_type="layer", bias=self.bias),
            InvertedResBlock(256, 256, expand_ratio=2, norm_type="layer", bias=self.bias),
            InvertedResBlock(256, 256, expand_ratio=2, norm_type="layer", bias=self.bias),
            ConvBlock(256, 128, kernel_size=3, stride=1, norm_type="layer", bias=self.bias),
        )

        self.upsample = nn.Sequential(
            UpConvLNormLReLU(128, 128),
            ConvBlock(128, 128, kernel_size=3, stride=1, norm_type="layer", bias=self.bias),
        )

        self.decoder = nn.Sequential(
            UpConvLNormLReLU(128, 64),
            ConvBlock(64, 64, kernel_size=3, stride=1, norm_type="layer", bias=self.bias),
            ConvBlock(64, 32, kernel_size=7, stride=1, norm_type="layer", bias=self.bias),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=self.bias),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, x):
        out = self.encoder(x)
        out = self.downsample(out)
        out = self.residual_blocks(out)
        out = self.upsample(out)
        img = self.decoder(out)

        return img
