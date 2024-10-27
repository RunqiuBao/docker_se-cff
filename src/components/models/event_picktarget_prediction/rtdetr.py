"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import importlib

import random 
import numpy as np 
from typing import List 

from .workspace import register
from .presnet import PResNet
from .hybrid_encoder import HybridEncoder
from .rtdetrv2_decoder import  RTDETRTransformerv2


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        rtdetr_cfg
    ):
        super().__init__()
            
        self._backbone = PResNet(**rtdetr_cfg['backbone'].params)
        self._decoder = RTDETRTransformerv2(**rtdetr_cfg['decoder'].params)
        self._encoder = HybridEncoder(**rtdetr_cfg['encoder'].params)

    def forward(self, x, targets=None):
        x = self._backbone(x)
        x = self._encoder(x)        
        x = self._decoder(x, targets)
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
