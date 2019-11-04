import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


print(th.__version__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.channels_x2 = nn.Conv2d(3, 6, 1)
    def forward(self, x):
        x = self.channels_x2(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

img1 = th.Tensor(np.ones((1, 3, 5, 6)))
print("Before: ", img1.shape)
print("After:  ", net(img1).shape)

for epoch in range(0):
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
