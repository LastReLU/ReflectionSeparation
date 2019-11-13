import os
import torchvision as thv
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch as th
import cv2
import random

MAX_INDOOR = 15000
MAX_OUTDOOR = 2700


class ImageDataSet(Dataset):
    def __init__(self, t_len=MAX_INDOOR, r_len=MAX_OUTDOOR, alpha=0.75, first_ref_pos=(0, 6), second_ref_pos=(6, 0), blur=5, random=True):
        self.t_path = './data/indoor'
        self.r_path = './data/outdoor'
        self.t_len = t_len
        self.r_len = r_len
        self.random = random

        self.alpha = alpha
        self.first_ref_pos = first_ref_pos
        self.second_ref_pos = second_ref_pos
        self.blur = blur

    def __random_prop(self, seed): # activate if random == True
        np.random.seed(seed)
        self.alpha = np.random.uniform(0.75, 0.8)
        np.random.seed(seed)
        pos = np.random.randint(7, size=4)
        self.first_ref_pos = (pos[0], pos[1])
        self.second_ref_pos = (pos[2], pos[3])
        np.random.seed(seed)
        self.blur = np.random.choice([1, 3, 5, 7, 9])

    def __len__(self):
        return self.t_len * self.r_len

    def __basic_crop(self, img, seed): # Image -> Image
        img_array = np.array(img)
        quot = min(img_array.shape[0], img_array.shape[1]) / 360
        if quot > 1:
            img = img.resize(
                (int(img_array.shape[1] / quot), int(img_array.shape[0] / quot)))
        random.seed(seed)
        crop = thv.transforms.RandomCrop(134)
        return crop(img)

    def __crop128(self, img):  # np.ndarray -> np.ndarray
        if img.shape[2] == 3:
            return img[3:131, 3:131]
        return img[:, 3:131, 3:131]

    def __get_img(self, id, path, seed):  # id (n, m, c) -> np.ndarray (c, n, m)
        img = Image.open('{}/{}.jpg'.format(path, id))
        img = img.convert('RGB')
        img = self.__basic_crop(img, seed)
        img = np.array(img)
        transposed_image = np.transpose(img, (2, 0, 1))
        return transposed_image

    def __bluring(self, img):  # np.ndarray [134x134] -> np.ndarray [128x128]
        return cv2.GaussianBlur(img, (self.blur, self.blur), 0)

    def __add_reflection(self, reflection_img): # np.ndarrays (c, n, m) -> int, np.ndarrays (c, n, m)
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
            reflection_img = th.Tensor(np.transpose(reflection_img, (0, 3, 1, 2)))
        if isinstance(reflection_img, np.ndarray):
            reflection_img = th.Tensor(reflection_img)

        reflection = F.conv2d(reflection_img, th.Tensor(kernel), groups=3)
        reflection = reflection.numpy().squeeze()

        return (1 - alpha1 - alpha2), reflection

    def __getitem__(self, id):
        if self.random:
            self.__random_prop(id)

        transition_img = self.__get_img(id // self.r_len, self.t_path, id)
        reflection_img = self.__get_img(id % self.r_len, self.r_path, id)

        transition_crop = self.__crop128(transition_img)
        reflection_blur = self.__bluring(reflection_img)

        k, reflection_layer = self.__add_reflection(reflection_blur)
        features_img = k*transition_crop + reflection_layer

        item = np.array([features_img, transition_crop, reflection_layer])
        item = th.Tensor(item / 255)
        #item = th.Tensor(item)
        return item

class DataLoader():
    def __init__(self, dataset, transition_num=1, reflection_num=18, seed=23, split=0.8, random=False, batch_size=4, test=False):
        self.dataset = dataset
        self.seed = seed
        self.transition_num = transition_num
        self.reflection_num = reflection_num
        self.random = random
        self.batch_size = batch_size

        t_split = int(dataset.t_len * split)
        r_split = int(dataset.r_len * split)

        if not test:
            np.random.seed(self.seed)
            self.transition_permutation = np.random.permutation(self.dataset.t_len)[:t_split]
            np.random.seed(self.seed)
            self.reflection_permutation = np.random.permutation(self.dataset.r_len)[:r_split]
        else:
            np.random.seed(self.seed)
            self.transition_permutation = np.random.permutation(self.dataset.t_len)[t_split:]
            np.random.seed(self.seed)
            self.reflection_permutation = np.random.permutation(self.dataset.r_len)[r_split:]

    def __len__(self):
        t_len = len(self.transition_permutation)
        r_len = len(self.reflection_permutation)
        if self.random:
            return min(t_len, r_len) // self.batch_size
        return t_len // self.transition_num

    def __get_batch(self, id):
        stack = []
        np.random.seed(id)
        r_split = np.random.randint(len(self.reflection_permutation) - self.reflection_num)
        for t_id in self.transition_permutation[id*self.transition_num : (id+1)*self.transition_num]:
            for r_id in self.reflection_permutation[r_split : r_split + self.reflection_num]:
                data_id = self.dataset.r_len * t_id + r_id
                stack.append(self.dataset[data_id])
        return stack

    def __get_random(self, id):
        stack = []
        for i in range(self.batch_size):
            t_id = self.transition_permutation[id*self.batch_size+i]
            r_id = self.reflection_permutation[id*self.batch_size+i]
            data_id = self.dataset.r_len * t_id + r_id
            stack.append(self.dataset[data_id])
        return stack

    def __getitem__(self, id):
        if self.random:
            batch = self.__get_random(id)
            if len(batch) < self.batch_size:
                raise StopIteration
        else:
            batch = self.__get_batch(id)
            if len(batch) < self.transition_num * self.reflection_num:
                raise StopIteration

        return th.stack(batch)
