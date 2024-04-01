# -*- coding: utf-8 -*-

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):

    def __init__(self, activation_type: str = 'relu'):
        super().__init__()
        self.activation_type = activation_type
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'gelu':
            self.activation = nn.GELU()
        elif activation_type == 'silu':
            self.activation = nn.SiLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'sin':
            self.activation = torch.sin

    def forward(self, x):
        return self.activation(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, first_act="relu", second_act="relu", **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.first_act = Activation(first_act)
        self.second_act = Activation(second_act)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.first_act(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.second_act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, first_act="relu", second_act="relu", third_act="relu", **kwargs):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.first_act = Activation(first_act)
        self.second_act = Activation(second_act)
        self.third_act = Activation(third_act)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.first_act(out)
        out = self.bn2(self.conv2(out))
        out = self.second_act(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.third_act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, act_rules=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if act_rules is None:
            act_rules = [
                {"first_act": torch.relu, "second_act": torch.relu},
                {"first_act": torch.relu, "second_act": torch.relu},
                {"first_act": torch.relu, "second_act": torch.relu},
                {"first_act": torch.relu, "second_act": torch.relu}
            ]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, first_act=act_rules[0]["first_act"], second_act=act_rules[0]["second_act"], third_act=act_rules[0].get("third_act", torch.relu))
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, first_act=act_rules[1]["first_act"], second_act=act_rules[1]["second_act"], third_act=act_rules[1].get("third_act", torch.relu))
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, first_act=act_rules[2]["first_act"], second_act=act_rules[2]["second_act"], third_act=act_rules[2].get("third_act", torch.relu))
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, first_act=act_rules[3]["first_act"], second_act=act_rules[3]["second_act"], third_act=act_rules[3].get("third_act", torch.relu))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, first_act, second_act, third_act):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, first_act=first_act, second_act=second_act, third_act=third_act))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(act_rules=None):
    return ResNet(BasicBlock, [2,2,2,2], act_rules=act_rules)

def ResNet34(act_rules=None):
    return ResNet(BasicBlock, [3,4,6,3], act_rules=act_rules)

def ResNet50(act_rules=None):
    return ResNet(Bottleneck, [3,4,6,3], act_rules=act_rules)

def ResNet101(act_rules=None):
    return ResNet(Bottleneck, [3,4,23,3], act_rules=act_rules)

def ResNet152(act_rules=None):
    return ResNet(Bottleneck, [3,8,36,3], act_rules=act_rules)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()