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

    # todo: split into train and test
    # todo: make evaluation dataloaders

    #model = model.DummyModel().to(device)
    model = torch.load('weights_new0.hdf5')
    model.eval()
    # let's fix the batch and iterate over it
    image_batch = cv2.imread("difficult/")
    batch = image_batch
    info = model.compute_all(batch, device=device)
    print(i, info)
