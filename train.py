import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
import data


hyper_params = {
    'indoor_size': 5,
    'outdoor_size': 25,
    'input_size': (3, 128, 128),
    'batch_size': 4,
    'num_epochs': 10,
    'learning_rate': 0.01
}


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
        [str(t) for t in Path("./data/indoor_row/").glob("*.jpg")],
        limit=100)
    outdoor_files = data.filter_images(
        [str(t) for t in Path("./data/outdoor_row/").glob("*.jpg")],
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
    # let's fix the batch and iterate over it
    for a, b in zip(trainloader_a, trainloader_b):
        batch = data.all_transform(a, b)
        break

    for step in range(1000):
        info = model.compute_all(batch, device=device)
        opt.zero_grad()
        info['loss'].backward()
        opt.step()

        # norm of grads for every weight
        for name, p in model.named_parameters():
            if 'weight' in name:
                train_writer.add_scalar("grad_" + name, np.linalg.norm(p.grad.data.cpu().numpy()), global_step=step)

        log.append(info['metrics'])
        print(info['metrics'])

        for k, v in batch.items():
            if k != 'alpha':
                for i in range(len(v)):
                    train_writer.add_image("{}_{}".format(k, i), v[0], global_step=step)

        for k, v in info['metrics'].items():
            train_writer.add_scalar(k, v, global_step=step)

        # todo: add evaluation loop
        model.eval()