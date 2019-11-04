import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NetArticle
ALPHA = 0.5




def train(train_loader_indoor, train_loader_outdoor, model, criterion, optimizer, epochs=0, alpha=ALPHA):
    losses = []
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(zip(train_loader_indoor, train_loader_outdoor)):
            batch_indoor, batch_outdoor = batch  # batch.shape ((bs, 3, n, n), (bs, 3, n, n)) ((bs, 3, n, n), (bs, 3, n, n))
            features_indoor, ground_truth_indoor = batch_indoor
            features_outdoor, ground_truth_outdoor = batch_outdoor
            features = alpha * features_indoor + (1 - alpha) * features_outdoor
            ground_truth = np.concatenate((ground_truth_indoor, ground_truth_outdoor), axis=1)

            optimizer.zero_grad()
            predicts = model(features)
            loss = criterion(predicts, ground_truth)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return losses


def test(test_loader, model, criterion, alpha=ALPHA):
    losses = []
    model.eval()
    for i, batch in enumerate(zip(test_loader_indoor, test_loader_outdoor)):
        batch_indoor, batch_outdoor = batch
        features_indoor, ground_truth_indoor = batch_indoor
        features_outdoor, ground_truth_outdoor = batch_outdoor
        features = alpha * features_indoor + (1 - alpha) * features_outdoor
        ground_truth = np.concatenate((ground_truth_indoor, ground_truth_outdoor), axis=1)

        predicts = model(features)
        loss = criterion(predicts, ground_truth)
        losses.append(loss.item())

    return losses


def main():
    #train_loader, test_loader -- работаем над этим
    net = NetArticle()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #train(train_loader, net, criterion, optimizer)
    #losses = test(test_loader, net, criterion)
    #print(losses)

    img1 = th.Tensor(np.ones((1, 3, 5, 6)))
    print("Before: ", img1.shape)
    print("After:  ", net(img1).shape)


print(th.__version__)
main()
