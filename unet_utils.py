import torch

"""
UNet Helper Functions
Authors: Rob Pitkin, Xinyang Zhao, Mona Ma

All code was derived from 
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

Copyright (C) 2015  Olaf Ronneberger, Philipp Fischer, Thomas Brox
"""


class ConvBlock(torch.nn.Module):
    """
    Convolutional block.
    * Performs two layers of 3x3 convolutions with specified number of filters.
    * Performs batch normalization after each conv layer and uses the
      ReLU activation function.
    * Can optionally include dropout with a given probability (layer
      inserted at the end of the block)
    """

    def __init__(self, num_filters=64, dropout_p=0):
        super().__init__()

        if dropout_p == 0:
            self.conv_block = torch.nn.Sequential(
                torch.nn.LazyConv2d(num_filters, kernel_size=3,
                                    padding='same'),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                torch.nn.LazyConv2d(num_filters, kernel_size=3,
                                    padding='same'),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU()
            )
        else:
            self.conv_block = torch.nn.Sequential(
                torch.nn.LazyConv2d(num_filters, kernel_size=3,
                                    padding='same'),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                torch.nn.LazyConv2d(num_filters, kernel_size=3,
                                    padding='same'),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_p)
            )

    def forward(self, x_input):
        return self.conv_block(x_input)


class DownBlock(torch.nn.Module):
    """
    Downsampling Block.
    * Performs a 2x2 MaxPooling operation on the input and then uses a
      ConvBlock.
    * Can optionally include dropout with a given probability (layer
      inserted at the end of the block)
    """

    def __init__(self, out_channels, dropout_p=0):
        super().__init__()

        self.down_block = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2),
            ConvBlock(out_channels, dropout_p),
        )

    def forward(self, x_input):
        return self.down_block(x_input)


class UpBlock(torch.nn.Module):
    """
    Upsampling Block.
    * Performs a transposed convolution on the input with a kernel size
      of 2 and a stride of 2 followed by a ConvBlock.
    * Can optionally include dropout with a given probability (layer
      inserted at the end of the block)
    """

    def __init__(self, out_channels, dropout_p=0):
        super().__init__()

        self.up = torch.nn.LazyConvTranspose2d(out_channels,
                                               kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels, dropout_p=dropout_p)

    def forward(self, x_input, skip):
        """
        Using code derived from
        https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        """
        x_input = self.up(x_input)
        # input is CHW
        diffY = skip.size()[2] - x_input.size()[2]
        diffX = skip.size()[3] - x_input.size()[3]

        x_input = torch.nn.functional.pad(x_input,
                                          [diffX // 2, diffX - diffX // 2,
                                           diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x_input], dim=1)
        return self.conv_block(x)


class OutBlock(torch.nn.Module):
    """
    Final output block. Converts the output size to the correct number
    of classes.
    """

    def __init__(self, out_channels):
        super(OutBlock, self).__init__()

        self.conv_layer = torch.nn.LazyConv2d(out_channels=out_channels,
                                              kernel_size=1, padding='same')

    def forward(self, x_input):
        return self.conv_layer(x_input)
