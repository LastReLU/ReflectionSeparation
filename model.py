import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2

class UnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.convs(x)


def save(imgs, bn, dir='train_out'):
    os.mkdir(f'data/{dir}/batch_{bn}')
    for i, img in enumerate(imgs):
        ready_img = np.maximum((np.transpose(img.detach().numpy(), (1, 2, 0))*255).astype(int), 0)
        cv2.imwrite(f'data/{dir}/batch_{bn}/img_{i}.jpg', ready_img)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.unet = UnetModel()
        self.restoration = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),
        )

    def forward(self, x):
        combined_features = self.stem(x)
        reflected_features = self.unet(combined_features)
        # what do you think about such a way?
        trans = self.restoration(combined_features - reflected_features)
        reflect = self.restoration(reflected_features)
        return dict(
            trans=trans,
            reflect=reflect,
        )

    def compute_all(self, batch, bn, device=None):
        #synthetic = batch['synthetic'].to(device)
        #alpha_transmitted = batch['alpha_transmitted'].to(device)
        #reflected = batch['reflected'].to(device)

        synthetic = batch['synthetic']
        alpha_transmitted = batch['alpha_transmitted']
        reflected = batch['reflected']

        output = self.forward(synthetic)

        loss_trans = F.mse_loss(output['trans'], alpha_transmitted)
        loss_refl = F.mse_loss(output['reflect'], reflected)

        #save(output['trans'], bn)

        loss = loss_refl + loss_trans
        # todo: add VGG L2
        return dict(
            loss=loss,
            metrics=dict(
                mse_trans=loss_trans.item(),
                mse_refl=loss_refl.item(),
            )
        )
