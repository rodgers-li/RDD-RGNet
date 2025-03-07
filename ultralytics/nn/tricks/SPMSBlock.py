# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

# from .dcn import deform_conv

##-----------------20230927 添加MSBlock-------------------##

## 参考<YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-time Object Detection>

from typing import Sequence, Union
from torch import Tensor

from ultralytics.nn.modules.conv import Conv

class MSBlockLayer(nn.Module):
    """MSBlockLayer

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int, tuple[int]): The kernel size of this Module.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to None.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int
                 ) -> None:
        super().__init__()
        self.in_conv = Conv(in_channel, out_channel, 1, 1)

        self.mid_conv = Conv(out_channel, out_channel, kernel_size, 1, g=out_channel)

        self.out_conv = Conv(out_channel, in_channel, 1, 1)


    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        x = self.in_conv(x)
        x = self.mid_conv(x)
        x = self.out_conv(x)
        return x


class MSBlock(nn.Module):
    """MSBlock

    Args:
        in_channel (int): The input channels of this Module.
        out_channel (int): The output channels of this Module.
        kernel_sizes (list(int, tuple[int])): Sequential of kernel sizes in MS-Block.

        in_expand_ratio (float): Channel expand ratio for inputs of MS-Block. Defaults to 3.
        mid_expand_ratio (float): Channel expand ratio for each branch in MS-Block. Defaults to 2.
        layers_num (int): Number of layer in MS-Block. Defaults to 3.
        in_down_ratio (float): Channel down ratio for downsample conv layer in MS-Block. Defaults to 1.

        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in MS-Block. Defaults to None.

        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 # kernel_sizes: Sequence[Union[int, Sequence[int]]],
                 kernel_sizes: Sequence[int],
                 layers_num: int = 1,
                 in_expand_ratio: float = 3.,
                 mid_expand_ratio: float = 2.,
                 in_down_ratio: float = 1.,

                 # attention_cfg: OptConfigType = None,
                 # conv_cfg: OptConfigType = None,
                 # norm_cfg: OptConfigType = dict(type='BN'),
                 # act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 ) -> None:
        super().__init__()

        self.in_channel = int(int(in_channel * in_expand_ratio) // in_down_ratio)
        self.mid_channel = self.in_channel // len(kernel_sizes)
        self.mid_expand_ratio = mid_expand_ratio
        groups = int(self.mid_channel * self.mid_expand_ratio)
        self.layers_num = layers_num
        self.in_attention = None

        self.attention = None
        # if attention_cfg is not None:
        #     attention_cfg["dim"] = out_channel
        #     self.attention = MODELS.build(attention_cfg)

        # self.in_conv = ConvModule(in_channel,
        #                           self.in_channel,
        #                           1,
        #                           conv_cfg=conv_cfg,
        #                           act_cfg=act_cfg,
        #                           norm_cfg=norm_cfg)

        self.in_conv = Conv(in_channel, self.in_channel, 1, 1)

        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(self.mid_channel,
                                      groups,
                                      kernel_size=kernel_size) for _ in range(int(self.layers_num))]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = Conv(self.in_channel, out_channel, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        out = self.in_conv(x)
        channels = []
        for i, mid_conv in enumerate(self.mid_convs):
            channel = out[:, i * self.mid_channel:(i + 1) * self.mid_channel, ...]
            if i >= 1:
                channel = channel + channels[i - 1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)
        return out


class SPMSBlock(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 # kernel_sizes: Sequence[Union[int, Sequence[int]]],
                 kernel_sizes: Sequence[int],
                 layers_num: int = 1,
                ) -> None:
        super().__init__()

        self.MSBlock_upper = MSBlock(in_channel, out_channel, [1,3], layers_num, in_expand_ratio=2., mid_expand_ratio=1., in_down_ratio= 1.,)
        # self.MSBlock_upper = Conv(in_channel, out_channel, 3, 1)
        self.MSBlock_lower = MSBlock(in_channel, out_channel, kernel_sizes, layers_num, in_expand_ratio=3., mid_expand_ratio=2., in_down_ratio= 1.,)
        self.cv = Conv(out_channel, out_channel, 3, 1)


    def forward(self, x):
        """Forward pass through C2f layer."""
        b, c, w, h = x.size()
        result = x.clone()
        x_upper = x[:, :, :, 0:int(h / 2)]
        x_lower = x[:, :, :, int(h / 2):h]
        y_upper = self.MSBlock_upper(x_upper)
        y_lower = self.MSBlock_upper(x_lower)

        result[:, :, :, 0:int(h / 2)] = y_upper
        result[:, :, :, int(h / 2):h] = y_lower

        result = self.cv(result)

        return result