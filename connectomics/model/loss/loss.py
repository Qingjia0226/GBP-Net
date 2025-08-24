from __future__ import print_function, division
from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
  
    def loss(self, pred, target, smooth=1, alpha=0.2, beta=0.8, gamma=2):
    

        loss = 0.0
        #print('1:',torch.unique(pred))
        for index in range(pred.size()[0]):
            inputs = pred[index].contiguous().view(-1)
            targets = target[index].contiguous().view(-1)
            TP = (inputs * targets).sum()
            FP = ((1 - targets) * inputs).sum()
            FN = (targets * (1 - inputs)).sum()
            Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
            
            loss += (1 - Tversky) ** gamma
        return loss#/pred.size()[0]

    def forward(self, pred, target, weight_mask=None):
        loss = self.loss(pred, target)
        return loss







