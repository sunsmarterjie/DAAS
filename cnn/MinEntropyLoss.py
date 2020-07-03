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

    def forward(self, input, target, alpha=0, beta=0, epoch=1):
        self.update(epoch)
        # loss1 = F.cross_entropy(input, target)
        loss1 = self.criterion(input, target)
        Alpha_normal = F.softmax(streng_func(alpha[0]).cuda(), dim=-1)
        normal_entLoss = torch.sum(torch.mul(Alpha_normal, torch.log(Alpha_normal)).cuda())
        Alpha_reduce = F.softmax(streng_func(alpha[1]).cuda(), dim=-1)
        reduce_entLoss = torch.sum(torch.mul(Alpha_reduce, torch.log(Alpha_reduce)).cuda())
        loss2 = (torch.add(-normal_entLoss, -reduce_entLoss))

        w = 0.2
        nor_dis_loss = 0
        nor_ent_loss = 0
        red_dis_loss = 0
        red_ent_loss = 0
        beta1 = beta[0]
        beta2 = beta[1]
        for i in range(3, 6):
            Beta = streng_func2(beta1[i - 3])
            nor_dis_loss += w * torch.pow((torch.sum(Beta[Beta > 0]) - 2), 2)
            Beta = F.softmax(Beta.cuda(), dim=-1)
            nor_ent_loss += torch.sum(torch.mul(Beta, torch.log(Beta)).cuda())
        for i in range(3, 6):
            Beta = streng_func2(beta2[i - 3])
            red_dis_loss += w * torch.pow((torch.sum(Beta[Beta > 0]) - 2), 2)
            Beta = F.softmax(Beta.cuda(), dim=-1)
            red_ent_loss += torch.sum(torch.mul(Beta, torch.log(Beta)).cuda())
        loss3 = (nor_dis_loss + red_dis_loss - nor_ent_loss - red_ent_loss)
        loss = loss1 + self.weight * self.weight1 * (self.weight2 * loss2 + 4 * loss3)
        return loss, self.weight * self.weight1 * self.weight2 * loss2, self.weight * self.weight1 * 4 * loss3

    def update(self, epoch):
        self.weight1 = linear(epoch)
        self.weight2 = log_(epoch)


def streng_func(t):
    x = 2 * t
    mask1 = (x < -1).float().cuda()
    mask2 = (x >= -1).float().cuda() + (x < 1).float().cuda() - 1
    mask3 = (x >= 1).float().cuda()
    x1 = torch.mul(mask1, torch.pow(x, 3))
    x2 = torch.mul(mask2, x)
    x3 = torch.mul(mask3, x)
    return x1 + x2 + x3
    # return t


def streng_func2(t):
    x = t
    mask1 = (x < 1).float().cuda()
    mask2 = (x >= 1).float().cuda()
    x1 = torch.mul(mask1, x).cuda()
    x2 = torch.mul(mask2, 1).cuda()
    return x1 + x2
