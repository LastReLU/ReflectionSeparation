import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NetArticle
import Image_DataSet as dtst
import cv2


hyper_params = {
    'indoor_size': 20,
    'outdoor_size': 100,
    'input_size': (3, 128, 128),
    'num_epochs': 10,
    'learning_rate': 0.001
}


def get_batch(batch):
    features = batch[:, 0, :, :, :]
    target_transpose = batch[:, 1, :, :, :]
    target_reflection = batch[:, 2, :, :, :]
    #target = th.Tensor(np.concatenate((target_transpose, target_reflection), axis=1))
    return features, target_transpose, target_reflection


def test(test_loader, model, criterion=cv2.PSNR):
    losses = []
    model.eval()
    step = 0
    for i, batch in enumerate(test_loader):
        features, target_transmission, target_reflection = get_batch(batch)
        predict_transmission, predict_reflection = model(features)
        print(predict_transmission[0].shape)
        if i < 5:
            print(target_transmission[0])
            #a = np.transpose(predict_transmission[0].detach().numpy(), (1, 2, 0))# * 255
            #a = predict_transmission[0]# * 255
            #print(a)
            cv2.imwrite("imgs/transmission" + str(i) + ".png", (np.transpose(predict_transmission[0].detach().numpy(), (1, 2, 0)) * 255).astype(int))
            cv2.imwrite("imgs/reflection" + str(i) + ".png", (np.transpose(predict_reflection[0].detach().numpy(), (1, 2, 0)) * 255).astype(int))
        loss1 = criterion(predict_transmission, target_transmission)
        loss2 = criterion(predict_reflection, target_reflection)
        loss = loss1 + loss2
        losses.append(loss)
        print(loss)
        step += 1
    print(sum(losses))


if __name__ == "__main__":
    print(th.__version__)

    data = dtst.ImageDataSet(hyper_params['indoor_size'], hyper_params['outdoor_size'])
    test_loader = dtst.DataLoader(data, 1, 18, test=True)

    #net = NetArticle()
    net = th.load("weights2.hdf5")
    losses = test(test_loader, net, criterion=nn.MSELoss())
    print(losses)

