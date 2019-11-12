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


def test(test_loader, model, criterion=cv2.PSNR, epochs=hyper_params["num_epochs"]):
    with experiment.train():
        losses = []
        model.eval()
        step = 0
        for i, batch in enumerate(test_loader):
            features, target_transmission, target_reflection = get_batch(batch)
            predict_transmission, predict_reflection = model(features)
            loss1 = criterion(predict_transmission, target_transmission)
            loss2 = criterion(predict_reflection, target_reflection)
            loss = loss1 + loss2
            experiment.log_metric("loss", loss, step=step)
            losses.append(loss)
            print(loss)
            step += 1
        print(sum(losses))

