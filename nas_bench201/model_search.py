import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import copy 
from utils import drop_path
from MinEntropyLoss import streng_func


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C):
        super(Cell, self).__init__()

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(1 + i):
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, weights):

        states = [s0]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return states[-1]


class Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(inplanes, planes, 1)
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=3, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        self.cells1 = nn.ModuleList()
        for i in range(layers):
            cell = Cell(steps, multiplier, C_curr)
            self.cells1 += [cell]
        self.down1 = Block(C_curr, C_curr * 2)
        C_curr *= 2
        self.cells2 = nn.ModuleList()
        for i in range(layers):
            cell = Cell(steps, multiplier, C_curr)
            self.cells2 += [cell]
        self.down2 = Block(C_curr, C_curr * 2)
        C_curr *= 2
        self.cells3 = nn.ModuleList()
        for i in range(layers):
            cell = Cell(steps, multiplier, C_curr)
            self.cells3 += [cell]

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_curr, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s = self.stem(input)
        for i, cell in enumerate(self.cells1):
            weights = F.softmax(streng_func(self.alphas_normal), dim=-1)
            s = cell(s, weights)
        s = self.down1(s)
        for i, cell in enumerate(self.cells2):
            weights = F.softmax(streng_func(self.alphas_normal), dim=-1)
            s = cell(s, weights)
        s = self.down2(s)
        for i, cell in enumerate(self.cells3):
            weights = F.softmax(streng_func(self.alphas_normal), dim=-1)
            s = cell(s, weights)
        out = self.global_pooling(s)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target, epoch):
        logits = self(input)
        return self._criterion(logits, target, alpha=self._arch_parameters, epoch=epoch)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(1 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-4 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
        ]

    def arch_parameters(self):
        return self._arch_parameters


if __name__ == '__main__':
    model = Network(16, 10, 8, criterion=None, steps=4, multiplier=4, stem_multiplier=3)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)
