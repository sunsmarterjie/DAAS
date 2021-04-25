import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dimension import *
import math
import copy
import torch.autograd.variable as V


class minEntropyLoss(nn.Module):
    def __init__(self, weight=0.1, interval=8):
        super(minEntropyLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss()
        self.interval = interval

    def forward(self, input, target, alpha=0, epoch=1):
        self.update(epoch)
        loss1 = self.criterion(input, target)
        Alpha_normal = F.softmax(streng_func(alpha[0]).cuda(), dim=-1)
        normal_entLoss = torch.sum(torch.mul(Alpha_normal, torch.log(Alpha_normal)).cuda())

        loss = loss1 - self.weight * normal_entLoss
        return loss, -self.weight * normal_entLoss

    def update(self, epoch):
        self.weight = linear(epoch - self.interval) / 2


def streng_func(t):
    x = 2 * t
    mask1 = (x < -1).float().cuda()
    mask2 = (x >= -1).float().cuda() + (x < 1).float().cuda() - 1
    mask3 = (x >= 1).float().cuda()
    x1 = torch.mul(mask1, torch.pow(x, 3))
    x2 = torch.mul(mask2, x)
    x3 = torch.mul(mask3, x)
    return x1 + x2 + x3
