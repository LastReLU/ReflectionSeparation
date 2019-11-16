import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2

import model
import data


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', default='./runs/0')
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--epochs', default=1, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device("cuda")

    indoor_size = 5400
    outdoor_size = indoor_size // 2
    multi_reflection = 16
    indoor_train_size = 4000
    outdoor_train_size = indoor_train_size // 2

    indoor_files = data.filter_images(
        [str(t) for t in Path("./data/indoor_row/").glob("*.jpg")],
        limit=indoor_size)
    outdoor_files = data.filter_images(
        [str(t) for t in Path("./data/outdoor_row/").glob("*.jpg")],
        limit=outdoor_size)
    indoor_files = np.array(multi_reflection * indoor_files)
    outdoor_files = np.array(2 * multi_reflection * outdoor_files)
    print("There are {} indoor and {} outdoor files".format(len(indoor_files), len(outdoor_files)))

    np.random.seed(228)
    indoor_train_ids = np.random.choice(multi_reflection * indoor_size, multi_reflection * indoor_train_size,
                                        replace=False)
    indoor_test_ids = ~np.isin(np.arange(multi_reflection * indoor_size), indoor_train_ids)
    outdoor_train_ids = np.random.choice(2 * multi_reflection * outdoor_size, 2 * multi_reflection * outdoor_train_size,
                                         replace=False)
    outdoor_test_ids = ~np.isin(np.arange(2 * multi_reflection * outdoor_size), outdoor_train_ids)

    _ = indoor_files[indoor_train_ids]
    indoor_test = indoor_files[indoor_test_ids]
    _ = outdoor_files[outdoor_train_ids]
    outdoor_test = outdoor_files[outdoor_test_ids]

    testloader_a = DataLoader(data.DummyDataset(indoor_test), batch_size=args.batch_size, shuffle=True, drop_last=True)
    testloader_b = DataLoader(data.DummyDataset(outdoor_test), batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = model.DummyModel().to(device)
    checkpoint = torch.load('lastrelu_v4_20_900.hdf5')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # let's fix the batch and iterate over it

    size_ = 0
    MSE = 0
    PSNR_sum = 0
    for i, (a, b) in enumerate(zip(testloader_a, testloader_b)):
        batch = data.all_transform(a, b)
        info = model.compute_all(batch, device=device)
        size_ += 1
        MSE += info['loss'].item()
        print(i, info['loss'].item())
        PSNR_sum += 20 * np.log10(1. / np.sqrt(info['loss'].item()))
    print('PSNR - ', PSNR_sum / size_, 'MSE - ', MSE / size_)
