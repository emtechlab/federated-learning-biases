import torch
import torch.nn as nn
import torch.nn.functional as F

# LayerNorm for 2D feature maps
class LayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm2d, self).__init__()
        # LayerNorm normalizes the last dimension, so we permute channels to the end
        self.layernorm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        # x: [N, C, H, W] -> [N, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        # back to [N, C, H, W]
        return x.permute(0, 3, 1, 2)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.ln1   = LayerNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.ln2   = LayerNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                LayerNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.ln1   = LayerNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.ln2   = LayerNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.ln3   = LayerNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                LayerNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = F.relu(self.ln2(self.conv2(out)))
        out = self.ln3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.ln1   = LayerNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)

class ResNet_AE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet_AE, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.ln1   = LayerNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        decoded = self.decoder(out1)

        out = F.relu(self.ln1(out1))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out, x, decoded

# Factory functions
def ResNet18():        return ResNet(BasicBlock, [2,2,2,2])
def ResNet18_AE():     return ResNet_AE(BasicBlock, [2,2,2,2])
def ResNet18_fer():    return ResNet(BasicBlock, [2,2,2,2], num_classes=7)
def ResNet18_cifar10():return ResNet(BasicBlock, [2,2,2,2], num_classes=10)
def ResNet18_utk():   return ResNet(BasicBlock, [2,2,2,2], num_classes=2)
def ResNet34():       return ResNet(BasicBlock, [3,4,6,3])
def ResNet50():       return ResNet(Bottleneck, [3,4,6,3])
def ResNet101():      return ResNet(Bottleneck, [3,4,23,3])
def ResNet152():      return ResNet(Bottleneck, [3,8,36,3])

def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
# test()
