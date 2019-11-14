import cv2
import os
import numpy as np


def save(imgs, dir='ti_ne_ukazal_put'):
    if not os.path.isdir('data/out'):
        os.mkdir('data/out')
    if not os.path.isdir(f'data/out/{dir}'):
        os.mkdir(f'data/out/{dir}')
    for i, img in enumerate(imgs):
        ready_img = np.maximum((np.transpose(img.detach().cpu().numpy(), (1, 2, 0)) * 255).astype(int), 0)
        cv2.imwrite(f'data/out/{dir}/img_{i}.jpg', ready_img)

