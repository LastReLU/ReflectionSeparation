import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NetArticle
import Image_DataSet as dtst

ALPHA = 0.9

def get_batch(batch):
    features = batch[:, 0, :, :, :]
    target_transpose = batch[:, 1, :, :, :]
    target_reflection = batch[:, 2, :, :, :]
    target = th.Tensor(np.concatenate((target_transpose, target_reflection), axis=1))
    return features, target

def train(train_loader, model, criterion, optimizer, epochs=10, alpha=ALPHA):
    losses = []
    model.train()
    for epoch in range(epochs):
        losses = []
        for i, batch in enumerate(train_loader):
            # batch.shape ((bs, 3, n, n), (bs, 3, n, n)) ((bs, 6, n, n), (bs, 6, n, n))
            features, target = get_batch(batch)
            optimizer.zero_grad()
            predicts = model(features)
            loss = criterion(predicts, target)
            loss.backward()
            optimizer.step()
            print(i, loss.item())
            losses.append(loss.item())
        print('batchend', sum(losses))
    return losses


def test(test_loader, model, criterion, alpha=ALPHA):
    losses = []
    model.eval()
    for i, batch in enumerate(test_loader):
        features, target = get_batch(batch)
        predicts = model(features)
        loss = criterion(predicts, target)
        losses.append(loss.item())

    return losses


def main():
    n = int(input('Indoor count: '))
    m = int(input('Outdoor count: '))

    data = dtst.ImageDataSet(n, m)

    train_loader = th.utils.data.DataLoader(data, batch_size=4)

    net = NetArticle()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    losses = train(train_loader, net, criterion, optimizer)
    #losses = test(test_loader, net, criterion)
    print(losses)

    img1 = th.Tensor(np.ones((1, 3, 128, 128)))
    print("Before: ", img1.shape)
    print("After:  ", net(img1).shape)


print(th.__version__)
main()
