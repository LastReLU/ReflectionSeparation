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
    p.add_argument('--batch_size', default=1, type=int)
    p.add_argument('--epochs', default=1, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device("cuda")

    indoor_files = data.filter_images(
        [str(t) for t in Path("./data/indoor_row/").glob("*.jpg")],
        limit=200)
    outdoor_files = data.filter_images(
        [str(t) for t in Path("./data/outdoor_row/").glob("*.jpg")],
        limit=200)
    print("There are {} indoor and {} outdoor files".format(len(indoor_files), len(outdoor_files)))

    # todo: split into train and test
    # todo: make evaluation dataloaders
    testloader_a = DataLoader(data.DummyDataset(indoor_files), batch_size=args.batch_size, shuffle=True, drop_last=True)
    testloader_b = DataLoader(data.DummyDataset(outdoor_files), batch_size=args.batch_size, shuffle=True, drop_last=True)

    #model = model.DummyModel().to(device)
    model = torch.load('lastrelu_v1_0_1000.hdf5')
    # let's fix the batch and iterate over it

    for i, (a, b) in enumerate(zip(testloader_a, testloader_b)):
        batch = data.all_transform(a, b)
        info = model.compute_all(batch, device=device)
        print(i, info)
