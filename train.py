import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NetArticle
import Image_DataSet as dtst
import Collate as clt

ALPHA = 0.9

def train(train_loader, lable, model, criterion, optimizer, epochs=10, alpha=ALPHA):
    losses = []
    model.train()
    for epoch in range(epochs):
        losses = []
        for i, (features, ground_truth) in enumerate(zip(train_loader, lable)):
            # batch.shape ((bs, 3, n, n), (bs, 3, n, n)) ((bs, 6, n, n), (bs, 6, n, n))
            optimizer.zero_grad()
            predicts = model(features)
            loss = criterion(predicts, ground_truth)
            loss.backward()
            optimizer.step()
            print(i, loss.item())
            losses.append(loss.item())
        print(sum(losses))
    return losses


def test(test_loader, lable, model, criterion, alpha=ALPHA):
    losses = []
    model.eval()
    for i, (features, ground_truth) in enumerate(zip(test_loader, lable)):
        predicts = model(features)
        loss = criterion(predicts, ground_truth)
        losses.append(loss.item())

    return losses


def main():
    n = 50

    data = dtst.ImageDataSet(n)

    train_loader = th.utils.data.DataLoader(data, batch_size=4, collate_fn=clt.CollateFn(alpha=ALPHA))
    lable_loader = th.utils.data.DataLoader(data, batch_size=4, collate_fn=clt.CollateFn(alpha=ALPHA, lable=True))

    net = NetArticle()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    losses = train(train_loader, lable_loader, net, criterion, optimizer)
    #losses = test(test_loader, net, criterion)
    print(losses)

    img1 = th.Tensor(np.ones((1, 3, 128, 128)))
    print("Before: ", img1.shape)
    print("After:  ", net(img1).shape)


print(th.__version__)
main()
