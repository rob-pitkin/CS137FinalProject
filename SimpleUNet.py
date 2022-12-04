from unet_utils import *

"""
Simple UNet Model Implementation
Authors: Rob Pitkin, Xinyang Zhao, Mona Ma

All code was derived from 
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

Copyright (C) 2015  Olaf Ronneberger, Philipp Fischer, Thomas Brox
"""


class UNetModel(torch.nn.Module):
    """
    Standard UNet Architecture from https://arxiv.org/pdf/1505.04597.pdf
    Modified to use 5 classes for the HuBMAP - Hacking the Human Body
    Kaggle challenge
    """
    def __init__(self, num_classes=5):
        super(UNetModel, self).__init__()

        self.conv_block = ConvBlock(64)
        self.down_block1 = DownBlock(128)
        self.down_block2 = DownBlock(256)
        self.down_block3 = DownBlock(512)
        self.down_block4 = DownBlock(1024)
        self.up_block4 = UpBlock(512)
        self.up_block3 = UpBlock(256)
        self.up_block2 = UpBlock(128)
        self.up_block1 = UpBlock(64)
        self.out_block = OutBlock(num_classes)

    def forward(self, x_input):
        x1 = self.conv_block(x_input)
        x2 = self.down_block1(x1)
        x3 = self.down_block2(x2)
        x4 = self.down_block3(x3)
        x5 = self.down_block4(x4)
        out = self.up_block4(x5, x4)
        out = self.up_block3(out, x3)
        out = self.up_block2(out, x2)
        out = self.up_block1(out, x1)
        out = self.out_block(out)
        return out
