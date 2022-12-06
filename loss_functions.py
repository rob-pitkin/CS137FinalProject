from typing import Optional, List

import torch
from segmentation_models_pytorch.losses import BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE
from segmentation_models_pytorch.losses._functional import to_tensor
from segmentation_models_pytorch.losses.dice import DiceLoss
from torch.nn.modules.loss import _Loss

"""
Loss functions to use for our model

Code inspired by https://arxiv.org/pdf/2006.14822.pdf and
the segmentation_models_pytorch library:
https://github.com/qubvel/segmentation_models.pytorch/
blob/master/segmentation_models_pytorch/losses/dice.py
"""


class LogCoshDiceLoss(_Loss):
    def __init__(
            self,
            mode: str,
            classes: Optional[List[int]] = None,
            log_loss: bool = False,
            from_logits: bool = True,
            smooth: float = 0.0,
            ignore_index: Optional[int] = None,
            eps: float = 1e-7,
    ):
        """
        Log-Cosh Dice loss wrapper.

        For parameter information and usage,
        see: https://github.com/qubvel/segmentation_models.pytorch/
             blob/master/segmentation_models_pytorch/losses/dice.py
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(LogCoshDiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self.dice = DiceLoss(mode, classes, log_loss,
                             from_logits, smooth, ignore_index, eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.torch.log(torch.cosh(torch.tensor(self.dice.forward(y_pred, y_true))))
