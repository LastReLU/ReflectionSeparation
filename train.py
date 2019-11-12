import numpy as np
from comet_ml import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NetArticle
import Image_DataSet as dtst


experiment = Experiment(api_key="3FVjB2xjP8kZyoywFZ6lLC5pC",
                        project_name="reflection-separation", workspace="wibbn")

hyper_params = {
    'indoor_size': 50,
    'outdoor_size': 50,
    'input_size': (3, 128, 128),
    'batch_size': 4,
    'num_epochs': 10,
    'learning_rate': 0.001
}

def get_batch(batch):
    features = batch[:, 0, :, :, :]
    target_transpose = batch[:, 1, :, :, :]
    target_reflection = batch[:, 2, :, :, :]
    #target = th.Tensor(np.concatenate((target_transpose, target_reflection), axis=1))
    return features, target_transpose, target_reflection


def train(train_loader, model, criterion, optimizer, epochs=hyper_params['num_epochs'], save=True):
    with experiment.train():
        losses = []
        model.train()
        step = 0
        for epoch in range(epochs):
            experiment.log_current_epoch(epoch)
            losses = []
            for i, batch in enumerate(train_loader):
                # batch.shape ((bs, 3, n, n), (bs, 3, n, n)) ((bs, 6, n, n), (bs, 6, n, n))
                features, target_transmission, target_reflection = get_batch(batch)
                optimizer.zero_grad()
                #predicts = model(features)
                predict_transmission, predict_reflection = model(features)
                #loss = criterion(predicts, target)
                loss1 = criterion(predict_transmission, target_transmission)
                loss2 = criterion(predict_reflection, target_reflection)
                loss = loss1 + loss2
                loss.backward()
                #print(loss.grad)
                optimizer.step()
                print(epoch, step, loss.item(), sum(sum(model.conv_dwon_6.weight.grad)))
                experiment.log_metric('loss', loss.item(), step=step)
                step += 1
                losses.append(loss.item())
                if save:
                    th.save(model, 'weights.hdf5')
            print('epoch end', sum(losses))
        return losses


if __init__ == "__main__":
    print(th.__version__)
    experiment.log_parameters(hyper_params)

    data = dtst.ImageDataSet(hyper_params['indoor_size'], hyper_params['outdoor_size'])
    train_loader = th.utils.data.DataLoader(data, batch_size=hyper_params['batch_size'])

    net = NetArticle()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hyper_params['learning_rate'])
    losses = train(train_loader, net, criterion, optimizer)
    #losses = test(test_loader, net, criterion)
    print(losses)

    img1 = th.Tensor(np.ones((1, 3, 128, 128)))
    print("Before: ", img1.shape)
    print("After:  ", net(img1).shape)
