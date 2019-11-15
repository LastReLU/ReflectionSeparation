import torch.nn as nn
import torch.nn.functional as F
import scripts


class UnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.convs(x)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
        )
        self.unet = UnetModel()
        self.restoration = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 9, padding=4),
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

    def compute_all(self, batch, device=None):
        synthetic = batch['synthetic'].to(device)
        alpha_transmitted = batch['alpha_transmitted'].to(device)
        reflected = batch['reflected'].to(device)
        output = self.forward(synthetic)

        loss_trans = F.mse_loss(output['trans'], alpha_transmitted)
        loss_refl = F.mse_loss(output['reflect'], reflected)

        loss = loss_refl + loss_trans

        scripts.save(output['trans'], 'imgs')
        scripts.save(batch['synthetic'], 'syns')
        scripts.save(batch['alpha_transmitted'], 'alphas')
        scripts.save(batch['reflected'], 'refs')
        # todo: add VGG L2
        return dict(
            loss=loss,
            metrics=dict(
                mse_trans=loss_trans.item(),
                mse_refl=loss_refl.item(),
            )
        )

    def predict(self, img, device=None):
        synthetic = img
        output = self.forward(synthetic)

        return dict(
            img_trans=output['trans'],
            img_refl=output['reflect']
        )
