import os
import torchvision as thv
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch as th
import cv2

MAX_INDOOR = 1000
MAX_OUTDOOR = 1000


class ImageDataSet(Dataset):
    def __init__(self, n_len=MAX_INDOOR, m_len=MAX_OUTDOOR, alpha=0.75, first_ref_pos=(0, 6), second_ref_pos=(6, 0), blur=5, random=True):
        self.indoor_path = './data/indoor'
        self.outdoor_path = './data/outdoor'
        self.n_len = n_len
        self.m_len = m_len
        self.random = random

        self.alpha = alpha
        self.first_ref_pos = first_ref_pos
        self.second_ref_pos = second_ref_pos
        self.blur = blur

    def __random_prop(self): # activate if random == True
        self.alpha = np.random.randint(75, 80 + 1) / 100
        self.first_ref_pos = (np.random.randint(7), np.random.randint(7))
        self.second_ref_pos = (np.random.randint(7), np.random.randint(7))
        self.blur = np.random.choice([1, 3, 5, 7, 9])

    def __len__(self):
        return self.n_len * self.m_len

    def __basic_crop(self, img): # Image -> Image
        img_array = np.array(img)
        quot = min(img_array.shape[0], img_array.shape[1]) / 360
        if quot > 1:
            img = img.resize(
                (int(img_array.shape[1] / quot), int(img_array.shape[0] / quot)))
        crop = thv.transforms.CenterCrop(134)
        return crop(img)

    def __crop128(self, img):  # np.ndarray -> np.ndarray
        if img.shape[2] == 3:
            return img[3:131, 3:131]
        return img[:, 3:131, 3:131]

    def __get_img(self, id, path):  # id (n, m, c) -> np.ndarray (c, n, m)
        img = Image.open('{}/{}.jpg'.format(path, id))
        img = img.convert('RGB')
        img = self.__basic_crop(img)
        img = np.array(img)
        transposed_image = np.transpose(img, (2, 0, 1))
        return transposed_image

    def __bluring(self, img):  # np.ndarray [134x134] -> np.ndarray [128x128]
        return cv2.GaussianBlur(img, (self.blur, self.blur), 0)

    def __add_reflection(self, transpose_img, reflection_img): # np.ndarrays (c, n, m) -> int, np.ndarrays (c, n, m)
        kernel_size = 7
        kernel = np.zeros((kernel_size, kernel_size))
        alpha1 = 1 - np.sqrt(self.alpha)
        alpha2 = np.sqrt(self.alpha) - self.alpha

        (x1, y1) = self.first_ref_pos
        (x2, y2) = self.second_ref_pos

        kernel[x1, y1] = alpha1
        kernel[x2, y2] = alpha2
        kernel = np.repeat(kernel[None, None, :, :], 3, 0)

        if len(reflection_img.shape) == 3:
            reflection_img = reflection_img[None, :, :, :]
        if reflection_img.shape[3] == 3:
            reflection_img = th.Tensor(
                np.transpose(reflection_img, (0, 3, 1, 2)))
        if isinstance(reflection_img, np.ndarray):
            reflection_img = th.Tensor(reflection_img)

        reflection = F.conv2d(reflection_img, th.Tensor(kernel), groups=3)
        reflection = reflection.numpy().squeeze()

        if len(transpose_img.shape) == 4:
            transpose_img = transpose_img.squeeze()

        return (1 - alpha1 - alpha2), transpose_img, reflection

    def __getitem__(self, id):
        if self.random:
            self.__random_prop()

        transpose_img = self.__get_img(id % self.n_len, self.indoor_path)
        reflection_img = self.__get_img(id // self.n_len, self.outdoor_path)

        transpose_crop = self.__crop128(transpose_img)
        reflection_blur = self.__bluring(reflection_img)

        k, transpose_layer, reflection_layer = self.__add_reflection(transpose_crop, reflection_blur)
        features_img = k*transpose_layer + reflection_layer

        item = np.array([features_img, transpose_crop, reflection_layer])
        item = th.Tensor(item / 255)
        return item