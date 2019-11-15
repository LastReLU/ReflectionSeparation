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
    p.add_argument('--batch_size', default=4, type=int)
    p.add_argument('--epochs', default=150, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    #device = torch.device("cuda")
    '''
    indoor_files = data.filter_images(
        [str(t) for t in Path("./data/indoor/").glob("*.jpg")],
        limit=100)
    outdoor_files = data.filter_images(
        [str(t) for t in Path("./data/outdoor/").glob("*.jpg")],
        limit=100)
    print("There are {} indoor and {} outdoor files".format(len(indoor_files), len(outdoor_files)))
    '''
    # todo: split into train and test
    # todo: make evaluation dataloaders
    indoor_size = 5400
    outdoor_size = indoor_size // 2
    multi_reflection = 16
    indoor_train_size = 4000
    outdoor_train_size = indoor_train_size // 2

    indoor_files = data.filter_images(
        [str(t) for t in Path("./data/indoor/").glob("*.jpg")],
        limit=indoor_size)
    outdoor_files = data.filter_images(
        [str(t) for t in Path("./data/outdoor/").glob("*.jpg")],
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

    indoor_train = indoor_files[indoor_train_ids]
    indoor_test = indoor_files[indoor_test_ids]
    outdoor_train = outdoor_files[outdoor_train_ids]
    outdoor_test = outdoor_files[outdoor_test_ids]

    trainloader_a = DataLoader(data.DummyDataset(indoor_train), batch_size=args.batch_size, shuffle=True, drop_last=True)
    trainloader_b = DataLoader(data.DummyDataset(outdoor_train), batch_size=args.batch_size, shuffle=True, drop_last=True)
    testloader_a = DataLoader(data.DummyDataset(indoor_test), batch_size=args.batch_size, shuffle=True, drop_last=True)
    testloader_b = DataLoader(data.DummyDataset(outdoor_test), batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = model.DummyModel()#.to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4)

    train_writer = SummaryWriter(args.logs)
    log = []

    model.train()
    for step in range(args.epochs):
        for i, (a, b) in enumerate(zip(trainloader_a, trainloader_b)):
            batch = data.all_transform(a, b)
            info = model.compute_all(batch)
            opt.zero_grad()
            info['loss'].backward()
            opt.step()

            print(step, i, info['metrics'])
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
            if step % 8 == 0 or step == 1:
                torch.save(model, 'weights_andrey_v2_' + str(step) + '.hdf5')
        # todo: add evaluation loop
    test()