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
from MinEntropyLoss import streng_func2


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

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        # for i in range(self._steps):
        #     s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
        #     offset += len(states)
        #     states.append(s)
        s2 = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
        offset += len(states)
        states.append(s2)
        s3 = sum(weights2[0][0][j] * 3.0 * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
        offset += len(states)
        states.append(s3)
        s4 = sum(weights2[1][0][j] * 4.0 * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
        offset += len(states)
        states.append(s4)
        s5 = sum(weights2[2][0][j] * 5.0 * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
        states.append(s5)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()
        self._initialize_betas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(streng_func(self.alphas_reduce), dim=-1)
                weights2 = list()
                weights2.append(F.softmax(streng_func2(self.beta_normal[0]), dim=-1))
                weights2.append(F.softmax(streng_func2(self.beta_normal[1]), dim=-1))
                weights2.append(F.softmax(streng_func2(self.beta_normal[2]), dim=-1))
            else:
                weights = F.softmax(streng_func(self.alphas_normal), dim=-1)
                weights2 = list()
                weights2.append(F.softmax(streng_func2(self.beta_reduce[0]), dim=-1))
                weights2.append(F.softmax(streng_func2(self.beta_reduce[1]), dim=-1))
                weights2.append(F.softmax(streng_func2(self.beta_reduce[2]), dim=-1))
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target, epoch):
        logits = self(input)
        return self._criterion(logits, target, alpha=self._arch_parameters, beta=self._beta_parameters, epoch=epoch)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-4 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-4 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def _initialize_betas(self):
        self.beta_normal0 = Variable(1e-4 * torch.randn(1, 3).cuda() + 0.666, requires_grad=True)
        self.beta_normal1 = Variable(1e-4 * torch.randn(1, 4).cuda() + 0.5, requires_grad=True)
        self.beta_normal2 = Variable(1e-4 * torch.randn(1, 5).cuda() + 0.4, requires_grad=True)
        self.beta_reduce0 = Variable(1e-4 * torch.randn(1, 3).cuda() + 0.666, requires_grad=True)
        self.beta_reduce1 = Variable(1e-4 * torch.randn(1, 4).cuda() + 0.5, requires_grad=True)
        self.beta_reduce2 = Variable(1e-4 * torch.randn(1, 5).cuda() + 0.4, requires_grad=True)
        # self.beta_normal0 = Variable(1e-4 * torch.randn(1, 3).cuda(), requires_grad=True)
        # self.beta_normal1 = Variable(1e-4 * torch.randn(1, 4).cuda(), requires_grad=True)
        # self.beta_normal2 = Variable(1e-4 * torch.randn(1, 5).cuda(), requires_grad=True)
        # self.beta_reduce0 = Variable(1e-4 * torch.randn(1, 3).cuda(), requires_grad=True)
        # self.beta_reduce1 = Variable(1e-4 * torch.randn(1, 4).cuda(), requires_grad=True)
        # self.beta_reduce2 = Variable(1e-4 * torch.randn(1, 5).cuda(), requires_grad=True)
        self.beta_normal = list([self.beta_normal0, self.beta_normal1, self.beta_normal2])
        self.beta_reduce = list([self.beta_reduce0, self.beta_reduce1, self.beta_reduce2])
        self._beta_parameters = [
            self.beta_normal,
            self.beta_reduce,
        ]
        self._beta_parameters_ = [
            self.beta_normal0,
            self.beta_normal1,
            self.beta_normal2,
            self.beta_reduce0,
            self.beta_reduce1,
            self.beta_reduce2,
        ]

    def beta_parameters(self):
        return self._beta_parameters_

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               # key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                               key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        # if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        def _sift_beta(betas, W):
            offset = 2
            node3 = sorted(range(len(betas[0][0])), key=lambda x: betas[0][0][x])
            node4 = sorted(range(len(betas[1][0])), key=lambda x: betas[1][0][x])
            node5 = sorted(range(len(betas[2][0])), key=lambda x: betas[2][0][x])
            W[offset + node3[0]][:] = 0
            offset += 3
            W[offset + node4[0]][:] = 0
            W[offset + node4[1]][:] = 0
            offset += 4
            W[offset + node5[0]][:] = 0
            W[offset + node5[1]][:] = 0
            W[offset + node5[2]][:] = 0
            return W

        alphas_normal = copy.deepcopy(self.alphas_normal)
        alphas_reduce = copy.deepcopy(self.alphas_reduce)
        alphas_normal = _sift_beta(self.beta_normal, alphas_normal)
        alphas_reduce = _sift_beta(self.beta_reduce, alphas_reduce)
        gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
