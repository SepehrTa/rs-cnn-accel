"""
resnet8.py - ResNet-8 for CIFAR-10, the MLPerf Tiny image-classification model.

    input 32x32x3
    stem:    conv3x3 16, BN, ReLU
    stack 1: [conv3x3 16, BN, ReLU, conv3x3 16, BN] + identity          -> ReLU   (32x32x16)
    stack 2: [conv3x3 32 /2, BN, ReLU, conv3x3 32, BN] + conv1x1 32 /2  -> ReLU   (16x16x32)
    stack 3: [conv3x3 64 /2, BN, ReLU, conv3x3 64, BN] + conv1x1 64 /2  -> ReLU   (8x8x64)
    global average pool -> fully connected 10

As in the MLPerf Tiny reference, the 1x1 shortcut convolutions have a bias and
no batch norm. About 78k parameters and 12.5M multiply-adds per image.

The parameter names used here are the contract with export (train.py) and the
NumPy models (graph.py): keep them in sync.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, 1, stride, 0, bias=True)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        s = x if self.shortcut is None else self.shortcut(x)
        return F.relu(y + s)


class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.block1 = Block(16, 16, 1)
        self.block2 = Block(16, 32, 2)
        self.block3 = Block(32, 64, 2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        x = self.block3(self.block2(self.block1(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
