from typing import Sequence

import torch.nn as nn
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets.unet import UNet
from torch import Tensor

from .layers import SkeletonizeLayer


class SkelUNet(UNet):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        pred_mask: bool = True,
    ) -> None:
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            channels,
            strides,
            kernel_size,
            up_kernel_size,
            num_res_units,
            act,
            norm,
            dropout,
            bias,
            adn_ordering,
        )
        self.pred_mask = pred_mask
        # input value range of SkeletonLayer: 0-1
        if out_channels == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
        self.skel_layer = SkeletonizeLayer()

    def forward(self, x: Tensor) -> Tensor:
        mask_pred = super().forward(x)
        skel_pred = self.activation(self.skel_layer(self.activation(mask_pred)))
        if self.pred_mask:
            return mask_pred, skel_pred
        else:
            return skel_pred

    def load_from(self, weights):
        self.load_state_dict(weights)
