import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NetArticle(nn.Module):
    def __init__(self):
        super(NetArticle, self).__init__()
        self.conv_intro_1 = nn.Conv2d(3, 16, kernel_size=9)
        self.conv_intro_2 = nn.Conv2d(16, 16, kernel_size=9)
        self.conv_intro_3456 = nn.Conv2d(16, 16, kernel_size=5)

        self.conv_down_1 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv_down_2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv_down_3 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv_down_4 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv_down_5 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv_down_6 = nn.Conv2d(128, 128, kernel_size=5)

        self.conv_up_1 = nn.Conv2d(128, 128, kernel_size=5)
        self.conv_up_2 = nn.Conv2d(128, 64, kernel_size=5)
        self.conv_up_3 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv_up_4 = nn.Conv2d(64, 32, kernel_size=5)
        self.conv_up_5 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv_up_6 = nn.Conv2d(32, 16, kernel_size=5)

        self.conv_final_1234 = nn.Conv2d(16, 16, kernel_size=5)
        self.conv_final_5 = nn.Conv2d(16, 16, kernel_size=9)
        self.conv_final_6 = nn.Conv2d(16, 3, kernel_size=9)

        self.channels_x2 = nn.Conv2d(3, 6, kernel_size=1)

    def forward(self, x):
        x = self.channels_x2(x)
        return x


class NetToy(nn.Module):
    def __init__(self):
        super(NetToy, self).__init__()
        self.channels_x2 = nn.Conv2d(3, 6, 1)
    def forward(self, x):
        x = self.channels_x2(x)
        return x
