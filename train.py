import numpy as np
from comet_ml import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NetArticle
import Image_DataSet as dtst

import config


experiment = Experiment(api_key=config.comet_ml_api,
                        project_name="reflection-separation", workspace="wibbn")

hyper_params = {
    'indoor_size': 20,
    'outdoor_size': 100,
    'input_size': (3, 128, 128),
    'batch_size': 4,
    'num_epochs': 10,
    'learning_rate': 0.001
}

def get_batch(batch):
    features = batch[:, 0, :, :, :]
    target_transpose = batch[:, 1, :, :, :]
    target_reflection = batch[:, 2, :, :, :]
    target = th.Tensor(np.concatenate((target_transpose, target_reflection), axis=1))
    return features, target

def train(train_loader, model, criterion, optimizer, epochs=hyper_params['num_epochs']):
    with experiment.train():
        losses = []
        model.train()
        step = 0
        for epoch in range(epochs):
            experiment.log_current_epoch(epoch)
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
                experiment.log_metric('loss', loss.item(), step=step)
                step += 1
                losses.append(loss.item())
            print('batchend', sum(losses))
        return losses


def test(test_loader, model, criterion):
    losses = []
    model.eval()
    for i, batch in enumerate(test_loader):
        features, target = get_batch(batch)
        predicts = model(features)
        loss = criterion(predicts, target)
        losses.append(loss.item())

    return losses


def main():
    experiment.log_parameters(hyper_params)

    data = dtst.ImageDataSet(hyper_params['indoor_size'], hyper_params['outdoor_size'])
    train_loader = dtst.DataLoader(data, 1, 18)

    net = NetArticle()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hyper_params['learning_rate'])
    losses = train(train_loader, net, criterion, optimizer)
    #losses = test(test_loader, net, criterion)
    print(losses)

    img1 = th.Tensor(np.ones((1, 3, 128, 128)))
    print("Before: ", img1.shape)
    print("After:  ", net(img1).shape)


print(th.__version__)
main()
