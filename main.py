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


def train(train_loader, model, criterion, optimizer, epochs=0):
    losses = []
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            features, ground_truth = batch
            optimizer.zero_grad()
            predicts = model(features)
            loss = criterion(predicts, featuress)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses


def test(test_loader, model, criterion):
    losses = []
    model.eval()
    for i, batch in enumerate(test_loader):
        features, ground_truth = batch
        predicts = model(features)
        loss = criterion(predicts, featuress)
        losses.append(loss.item())
    return losses


#train_loader, test_loader -- работаем над этим
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#train(train_loader, net, criterion, optimizer)
#losses = test(test_loader, net, criterion)
#print(losses)

img1 = th.Tensor(np.ones((1, 3, 5, 6)))
print("Before: ", img1.shape)
print("After:  ", net(img1).shape)

