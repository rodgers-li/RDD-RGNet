# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .SPMSBlock import SPMSBlock

from .attention import binary_spatial_Attention, CoordAtt, SE_Att, ECA_Att

from .CARAFE import CARAFE

from .DeformableConv import DCNv2

__all__ = ('SPMSBlock', 'DSConv', 'DSConv_C2f', 'SAConv', 'DSConv_block', 'iRMB',
           'CoordAtt', 'binary_spatial_Attention', 'CARAFE', 'DCNv2',  'SE_Att', 'ECA_Att' )

