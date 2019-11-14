import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
import data


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', default='./runs/0')
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--epochs', default=10, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device("cuda")

    indoor_files = data.filter_images(
        [str(t) for t in Path("./data/indoor/").glob("*.jpg")],
        limit=100)
    outdoor_files = data.filter_images(
        [str(t) for t in Path("./data/outdoor/").glob("*.jpg")],
        limit=100)
    print("There are {} indoor and {} outdoor files".format(len(indoor_files), len(outdoor_files)))

    # todo: split into train and test
    # todo: make evaluation dataloaders
    trainloader_a = DataLoader(data.DummyDataset(indoor_files), batch_size=args.batch_size, shuffle=True, drop_last=True)
    trainloader_b = DataLoader(data.DummyDataset(outdoor_files), batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = model.DummyModel().to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4)

    train_writer = SummaryWriter(args.logs)
    log = []

    model.train()
    for step in range(1000):
        for i, (a, b) for enumerate(zip(traainloader_a, trainloader_b)):
            batch = data.all_transform(a, b)
            info = model.compute_all(batch, device=device)
            opt.zero_grad()
            info['loss'].backward()
            opt.step()

            # norm of grads for every weight
            '''
            for name, p in model.named_parameters():
                if 'weight' in name:
                    train_writer.add_scalar("grad_" + name, np.linalg.norm(p.grad.data.cpu().numpy()), global_step=step)

            log.append(info['metrics'])
            print(step, info['metrics'])

            for k, v in batch.items():
                if k != 'alpha':
                    for i in range(len(v)):
                        train_writer.add_image("{}_{}".format(k, i), v[0], global_step=step)

            for k, v in info['metrics'].items():
                train_writer.add_scalar(k, v, global_step=step)
            '''
            torch.save(model, 'weights_new1.hdf5')
        # todo: add evaluation loop
        model.eval()

