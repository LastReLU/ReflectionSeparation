import numpy as np
#from comet_ml import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NetArticle
import Image_DataSet as dtst
from torch.utils.data import DataLoader
import constant_DataSet as cds
from PIL import Image

#import config


#experiment = Experiment(api_key=config.comet_ml_api,
#                        project_name="reflection-separation", workspace="wibbn")

hyper_params = {
    'indoor_size': 5,
    'outdoor_size': 25,
    'input_size': (3, 128, 128),
    'batch_size': 1,
    'num_epochs': 2,
    'learning_rate': 0.01
}

def get_batch(batch):
    features = batch[:, 0, :, :, :]
    target_transpose = batch[:, 1, :, :, :]
    target_reflection = batch[:, 2, :, :, :]
    #target = th.Tensor(np.concatenate((target_transpose, target_reflection), axis=1))
    return features, target_transpose, target_reflection


def train(train_loader, model, criterion, optimizer, epochs=hyper_params['num_epochs'], save=True, device=th.device("cuda")):
    #with experiment.train():
    losses = []
    model.train()
    step = 0
    for epoch in range(epochs):
        #experiment.log_current_epoch(epoch)
        losses = []
        for i, batch in enumerate(train_loader):
            features, target_transmission = batch[:, :3, :, :], batch[:, 3:, :, :]
            # print('featers and target', features[0, :, : , :], target_transmission[0, :, :, :])
            #features = features.to(device)
            #target_transmission = target_transmission.to(device)
            optimizer.zero_grad()

            #predict_transmission, predict_reflection = model(features)
            predict_transmission = model(features)
            loss = criterion(predict_transmission, target_transmission)

            """
            Two head version
            #loss2 = criterion(predict_reflection, target_reflection)
            #loss = loss1 + loss2
            #print("LOSSES: ", loss1, loss)
            """
            loss.backward()
            optimizer.step()
            # print(epoch, step, loss.item())

            #if (loss.item() > 1) :
            # Image.fromarray((np.transpose(features[0].numpy(), (1, 2, 0)) * 255).astype('uint8')).show()
            # Image.fromarray((np.transpose(target_transmission[0].numpy(), (1, 2, 0)) * 255).astype('uint8')).show()
            # Image.fromarray((np.transpose(predict_transmission[0].detach().numpy(), (1, 2, 0)) * 255).astype('uint8')).show()

            print('epoch number is ', epoch, '\n', "step number is ", step, '\n', 'loss ', loss.item(), '\n',
                  th.mean(model.conv_head1_1.weight.grad[0][0]), '\n', th.mean(model.conv_head1_2.weight.grad[0][0]), '\n',
                  th.mean(model.conv_head1_3.weight.grad[0][0]), '\n', th.mean(model.conv_head1_4.weight.grad[0][0]), '\n',
                  th.mean(model.conv_head1_5.weight.grad[0][0]), '\n', th.mean(model.conv_head1_6.weight.grad[0][0]))
            print()
            #exit()
            #experiment.log_metric('loss', loss.item(), step=step)
            step += 1
            losses.append(loss.item())
            if save:
                th.save(model, 'weights3.hdf5')
        print('epoch end', sum(losses))
    return losses


if __name__ == "__main__":
    print(th.__version__)
    #experiment.log_parameters(hyper_params)
    #device = th.device("cuda" if th.cuda().is_available() else "cpu")
    #device = th.device("cuda")
    #print(device)
    #data = dtst.ImageDataSet(hyper_params['indoor_size'], hyper_params['outdoor_size'])
    #train_loader = dtst.DataLoader(data, 1, 18)

    train_dataloader = DataLoader(cds.const_DataSet(test=False), batch_size=4)

    net = NetArticle()
    #net = th.load("weights2.hdf5")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hyper_params['learning_rate'])
    #optimizer = optim.SGD(net.parameters(), lr=hyper_params['learning_rate'])
    losses = train(train_dataloader, net, criterion, optimizer)
    print(losses)

