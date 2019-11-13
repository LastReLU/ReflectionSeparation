import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NetArticle(nn.Module):
    def __init__(self):
        super(NetArticle, self).__init__()
        self.conv_intro_1 = nn.Conv2d(3, 32, kernel_size=9, padding=4)
        self.conv_intro_2 = nn.Conv2d(32, 32, kernel_size=9, padding=4)
        self.conv_intro_3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_intro_4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_intro_5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_intro_6 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        self.conv_down_1 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_down_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2) # 32 to concat
        self.conv_down_3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_down_4 = nn.Conv2d(32, 32, kernel_size=5, padding=2) # 64 to concat
        self.conv_down_5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_down_6 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        self.conv_up_1 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_up_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_up_3 = nn.Conv2d(32, 32, kernel_size=5, padding=2) # not 64. but 64 + 64 = 128 because of concat
        self.conv_up_4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_up_5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)  # not 32. but 32 + 32 = 64 because of concat
        self.conv_up_6 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        self.conv_head1_1 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_head1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_head1_3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_head1_4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv_head1_5 = nn.Conv2d(32, 32, kernel_size=9, padding=4)
        self.conv_head1_6 = nn.Conv2d(32, 3, kernel_size=9, padding=4) # not 16, 3 as two first layers. but 16, 6 because of concat

        self.conv_head2_1 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv_head2_2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv_head2_3 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv_head2_4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv_head2_5 = nn.Conv2d(16, 16, kernel_size=9, padding=4)
        self.conv_head2_6 = nn.Conv2d(16, 3, kernel_size=9, padding=4) # not 16, 3 as two first layers. but 16, 6 because of concat

    def intro(self, x):
        x = self.conv_intro_1(x)
        #x = F.relu(x)
        x = self.conv_intro_2(x)
        #x = F.relu(x)
        x = self.conv_intro_3(x)
        x = F.relu(x)
        x = self.conv_intro_4(x)
        #x = F.relu(x)
        x = self.conv_intro_5(x)
        #x = F.relu(x)
        x = self.conv_intro_6(x)
        x = F.relu(x)
        return x

    def body(self, x):
        legacy1 = self.conv_down_1(x)
        #legacy1 = F.relu(legacy1)
        legacy1 = self.conv_down_2(legacy1)
        legacy1 = F.relu(legacy1)   # shape is 32
        legacy2 = self.conv_down_3(legacy1)
        #legacy2 = F.relu(legacy2)
        legacy2 = self.conv_down_4(legacy2)
        legacy2 = F.relu(legacy2)   # shape is 64
        legacy3 = self.conv_down_5(legacy2)
        #legacy3 = F.relu(legacy3)
        legacy3 = self.conv_down_6(legacy3)
        legacy3 = F.relu(legacy3)   # shape is 128 (actually we needn't legacy3)
        up = self.conv_up_1(legacy3)
        #up = F.relu(up)
        up = self.conv_up_2(up)
        up = F.relu(up)
        up = self.conv_up_3(legacy2 + up)
        #up = F.relu(up)
        up = self.conv_up_4(up)
        up = F.relu(up)
        up = self.conv_up_5(legacy1 + up)
        #up = F.relu(up)
        up = self.conv_up_6(up)
        up = F.relu(x - up)
        return up

    def head1(self, x):
        x = self.conv_head1_1(x)
        #x = F.relu(x)
        x = self.conv_head1_2(x)
        x = F.relu(x)
        x = self.conv_head1_3(x)
        #x = F.relu(x)
        x = self.conv_head1_4(x)
        x = F.relu(x)
        x = self.conv_head1_5(x)
        #x = F.relu(x)
        x = self.conv_head1_6(x)
        x = F.relu(x)
        return x

    def head2(self, x):
        x = self.conv_head2_1(x)
        x = F.relu(x)
        x = self.conv_head2_2(x)
        x = F.relu(x)
        x = self.conv_head2_3(x)
        x = F.relu(x)
        x = self.conv_head2_4(x)
        x = F.relu(x)
        x = self.conv_head2_5(x)
        x = F.relu(x)
        x = self.conv_head2_6(x)
        x = F.relu(x)
        return x

    def forward(self, x):
        intro_end = self.intro(x)
        body_end = self.body(intro_end)
        transmission = self.head1(body_end)
        #reflection = self.head2(body_end)
        return transmission#, reflection

