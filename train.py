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
    'indoor_size': 5,
    'outdoor_size': 25,
    'input_size': (3, 128, 128),
    'batch_size': 4,
    'num_epochs': 10,
    'learning_rate': 0.01
}

def get_batch(batch):
    features = batch[:, 0, :, :, :]
    target_transpose = batch[:, 1, :, :, :]
    target_reflection = batch[:, 2, :, :, :]
    #target = th.Tensor(np.concatenate((target_transpose, target_reflection), axis=1))
    return features, target_transpose, target_reflection


def train(train_loader, model, criterion, optimizer, epochs=hyper_params['num_epochs'], save=True, device=th.device("cuda")):
    with experiment.train():
        losses = []
        model.train()
        step = 0
        for epoch in range(epochs):
            experiment.log_current_epoch(epoch)
            losses = []
            for i, batch in enumerate(train_loader):
                features, target_transmission, target_reflection = get_batch(batch)
                features = features.to(device)
                target_transmission = target_transmission.to(device)
                optimizer.zero_grad()
                #predict_transmission, predict_reflection = model(features)
                predict_transmission = model(features)
                print('__________________________________')
                #print(predict_transmission[0])
                #print(target_transmission[0] - predict_transmission[0])
                loss = criterion(predict_transmission, target_transmission)
                #loss2 = criterion(predict_reflection, target_reflection)
                #loss = loss1 + loss2
                #print("LOSSES: ", loss1, loss)
                loss.backward()
                optimizer.step()
                print(epoch, step, loss.item(), th.mean(model.conv_down_6.weight.grad[0][0]), th.mean(model.conv_intro_2.weight.grad[0][0]))
                experiment.log_metric('loss', loss.item(), step=step)
                step += 1
                losses.append(loss.item())
                if save:
                    th.save(model, 'weights3.hdf5')
            print('epoch end', sum(losses))
        return losses


if __name__ == "__main__":
    print(th.__version__)
    experiment.log_parameters(hyper_params)
    #device = th.device("cuda" if th.cuda().is_available() else "cpu")
    device = th.device("cuda")
    print(device)
    data = dtst.ImageDataSet(hyper_params['indoor_size'], hyper_params['outdoor_size'])
    train_loader = dtst.DataLoader(data, 1, 18)

    net = NetArticle().to(device)
    #net = th.load("weights2.hdf5")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hyper_params['learning_rate'])
    #optimizer = optim.SGD(net.parameters(), lr=hyper_params['learning_rate'])
    losses = train(train_loader, net, criterion, optimizer)
    print(losses)

