import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .layers import get_norm
from lib.common import initialize_weights


class Discriminator(nn.Module):
    def __init__(self, dataset_name: str, num_layers: int = 1, use_spectral_norm: bool = False, normalization: str = "instance"):
        """
        Initialize the Discriminator model.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        num_layers : int, optional
            Number of layers in the discriminator
        """
        super().__init__()
        self.name = f"discriminator_{dataset_name}"
        self.bias = False
        initial_channels = 32

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1, bias=self.bias))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        in_channels = initial_channels
        for _ in range(num_layers):
            self.layers.append(nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1, bias=self.bias))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.layers.append(nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, stride=1, padding=1, bias=self.bias))
            self.layers.append(get_norm(normalization, in_channels * 4))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels *= 4

        self.layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=self.bias))
        self.layers.append(get_norm(normalization, in_channels))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers.append(nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=self.bias))

        if use_spectral_norm:
            self.layers = [spectral_norm(layer) if isinstance(layer, nn.Conv2d) else layer for layer in self.layers]

        self.discriminator_module = nn.Sequential(*self.layers)
        initialize_weights(self)

    def discriminate(self, img):
        return self.discriminator_module(img)

    def forward(self, img):
        logits = self.discriminate(img)
        return logits
