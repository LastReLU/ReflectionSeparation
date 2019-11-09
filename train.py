import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NetArticle
from PIL import Image
import multiDataSet as mrds
ALPHA = 0.9

def train(data_loader, model, criterion, optimizer, epochs=2, alpha=ALPHA):
    losses = []
    model.train()
    for epoch in range(epochs):
        losses = []
        for i, (features, labels) in enumerate(data_loader):
            #if (i == 0):
            #    print(features.shape, labels.shape)
            optimizer.zero_grad()
            predicts = model(features)
            loss = criterion(predicts, labels)
            loss.backward()
            optimizer.step()
            print(i, loss.item())
            losses.append(loss.item())

        #print(sum(losses))
    return losses


def test(data_loader, model, criterion, alpha=ALPHA):
    losses = []
    model.eval()
    for i, (features, labels) in enumerate(data_loader):
        predicts = model(features)
        loss = criterion(predicts, labels)
        losses.append(loss.item())

    return losses


def main():
    train_loader = mrds.MultiReflectionDataSet(5, 5, 4, True)
    test_loader = mrds.MultiReflectionDataSet(3, 3, 2, False)

    net = NetArticle()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    _ = train(train_loader, net, criterion, optimizer)
    losses = test(test_loader, net, criterion)
    print(losses)

    img1 = th.Tensor(np.ones((1, 3, 128, 128)))
    #print("Before: ", img1.shape)
    #print("After:  ", net(img1).shape)

print(th.__version__)
main()
