# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["mynet"]


class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = self.conv1_bn(out)
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv_2(out))
        out = self.conv2_bn(out)
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


def mynet(num_classes):
    return MyNet(num_classes=num_classes)
