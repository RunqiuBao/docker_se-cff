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
from .matcher import HungarianMatcher
from .rtdetr_criterion import RTDETRCriterion


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        rtdetr_cfg,
        loss_cfg,
        **kwargs
    ):
        super().__init__()
            
        self._backbone = PResNet(**rtdetr_cfg['backbone'].params)
        self._decoder = RTDETRTransformerv2(**rtdetr_cfg['decoder'].params)
        self._encoder = HybridEncoder(**rtdetr_cfg['encoder'].params)

        matcher = HungarianMatcher(**loss_cfg.matcher.params)
        self.loss_functor = RTDETRCriterion(matcher, **loss_cfg.rtdetr_loss)

    def forward(self, x, targets=None):
        x = self._backbone(x)
        x = self._encoder(x)        
        x = self._decoder(x, targets)
        return x
    
    def forward_ext(self, x, x_right, targets=None):
        x = self._backbone(x)
        x_right = self._backbone(x_right)
        x = self._encoder(x)        
        x = self._decoder(x, targets)
        return x, x_right

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
    
    def compute_loss(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs):
        if self.training:
            self.loss_functor.train()
        else:
            self.loss_functor.eval()
        (
            loss_leftdetections,
            selected_leftdetections,
            corresponding_gt_labels,
            indices
        ) = self.loss_functor(
            preds,
            labels,
            **kwargs
        )

        return loss_leftdetections, (selected_leftdetections, corresponding_gt_labels, indices)
        

