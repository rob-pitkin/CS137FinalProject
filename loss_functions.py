import torch

"""
Loss functions to use for our model

Code inspired by https://arxiv.org/pdf/2006.14822.pdf
"""


def dice_loss(output, target):
    """
    Dice Loss function based on the DICE Coefficient

    :param output: The predicted output
    :param target: The target label mask
    :return: The calculated Dice Loss
    """
    loss = 1 - ((2 * torch.matmul(target, output) + 1) / (torch.add(target, output) + 1))
    return loss


def log_cosh_dice_loss(output, target):
    """
    Log-Cosh Dice Loss. Ensures the loss is tractable.

    :param output: The predicted output
    :param target: The target label mask
    :return: The calculated Log-Cosh Dice Loss
    """
    loss = torch.log(torch.cosh(torch.tensor(dice_loss(output, target))))
    return loss
