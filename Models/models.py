import scipy.io as io
import torch.nn as nn
import torchvision
from segmentation_models_pytorch import Unet
import torch

from model_blocks import *


class Generator(nn.Module):
    def __init__(self, config, num_of_classes):
        super(Generator, self).__init__()
        self.config = config
        self.unet = Unet(encoder_name='resnet18', encoder_weights='imagenet', in_channels=3, classes=num_of_classes,
                         activation='sigmoid', decoder_use_batchnorm='inplace')

    def forward(self, x):
        return self.unet(x)


class Discriminator(nn.Module):
    """
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake.
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    """
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn
