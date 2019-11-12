import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NetArticle(nn.Module):
    def __init__(self):
        super(NetArticle, self).__init__()
        self.conv_intro_1 = nn.Conv2d(3, 16, kernel_size=9, padding=4)
        self.conv_intro_2 = nn.Conv2d(16, 16, kernel_size=9, padding=4)
        self.conv_intro_3456 = nn.Conv2d(16, 16, kernel_size=5, padding=2)

        self.conv_down_1 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv_down_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2) # 32 to concat
        self.conv_down_3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv_down_4 = nn.Conv2d(64, 64, kernel_size=5, padding=2) # 64 to concat
        self.conv_down_5 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv_down_6 = nn.Conv2d(128, 128, kernel_size=5, padding=2)

        self.conv_up_1 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.conv_up_2 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.conv_up_3 = nn.Conv2d(64, 64, kernel_size=5, padding=2) # not 64. but 64 + 64 = 128 because of concat
        self.conv_up_4 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv_up_5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)  # not 32. but 32 + 32 = 64 because of concat
        self.conv_up_6 = nn.Conv2d(32, 16, kernel_size=5, padding=2)

        self.conv_final_1234 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv_final_5 = nn.Conv2d(16, 16, kernel_size=9, padding=4)
        self.conv_final_6 = nn.Conv2d(16, 3, kernel_size=9, padding=4) # not 16, 3 as two first layers. but 16, 6 because of concat

        #self.conv_1x1 = nn.Conv2d(3, 3, kernel_size=1)

    def intro(self, x):
        x = self.conv_intro_1(x)
        x = F.relu(x)
        x = self.conv_intro_2(x)
        x = F.relu(x)
        x = self.conv_intro_3456(x)
        x = F.relu(x)
        x = self.conv_intro_3456(x)
        x = F.relu(x)
        x = self.conv_intro_3456(x)
        x = F.relu(x)
        x = self.conv_intro_3456(x)
        x = F.relu(x)
        return x

    def body(self, x):
        legacy1 = self.conv_down_1(x)
        legacy1 = F.relu(legacy1)
        legacy1 = self.conv_down_2(legacy1)
        legacy1 = F.relu(legacy1)   # shape is 32
        legacy2 = self.conv_down_3(legacy1)
        legacy2 = F.relu(legacy2)
        legacy2 = self.conv_down_4(legacy2)
        legacy2 = F.relu(legacy2)   # shape is 64
        legacy3 = self.conv_down_5(legacy2)
        legacy3 = F.relu(legacy3)
        legacy3 = self.conv_down_6(legacy3)
        legacy3 = F.relu(legacy3)   # shape is 128 (actually we needn't legacy3)
        up = self.conv_up_1(legacy3)
        up = F.relu(up)
        up = self.conv_up_2(up)
        up = F.relu(up)
        up = self.conv_up_3(legacy2 + up)
        up = F.relu(up)
        up = self.conv_up_4(up)
        up = F.relu(up)
        up = self.conv_up_5(legacy1 + up)
        up = F.relu(up)
        up = self.conv_up_6(up)
        up = F.relu(up)
        return up

    def head12(self, x):
        x = self.conv_final_1234(x)
        x = F.relu(x)
        x = self.conv_final_1234(x)
        x = F.relu(x)
        x = self.conv_final_1234(x)
        x = F.relu(x)
        x = self.conv_final_1234(x)
        x = F.relu(x)
        x = self.conv_final_5(x)
        x = F.relu(x)
        x = self.conv_final_6(x)
        x = F.relu(x)
        return x

    def forward(self, x):
        x = self.intro(x)
        x = self.body(x)
        transmission = self.head12(x)
        reflection = self.head12(x)
        return transmission, reflection

