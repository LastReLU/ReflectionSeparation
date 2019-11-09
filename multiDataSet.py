import os
import torchvision as thv
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch as th
import torch.nn.functional as F
from scipy.misc import imresize # pip install scipy==1.2.0
import cv2
import random

INDOOR_SIZE, OUTDOOR_SIZE = 2303, 205
all_permutation_indoor = np.random.permutation(INDOOR_SIZE)
all_permutation_outdoor = np.random.permutation(OUTDOOR_SIZE)

class MultiReflectionDataSet(Dataset):
    def __init__(self, indoor_size=10, outdoor_size=10, batch_size=4, train=True):
        self.train = train
        self.indoor_size = indoor_size
        self.outdoor_size = outdoor_size
        self.permutation_indoor = np.random.permutation(self.indoor_size)
        self.permutation_outdoor = np.random.permutation(self.outdoor_size)
        self.indoor_path = './data/indoor'
        self.outdoor_path = './data/outdoor'
        self.batch_size = batch_size
        self.id_indoor = 0

    def __len__(self):
        return self.indoor_size

    def __get_img(self, id, path, is_reflection):
        img = ''
        if path == self.indoor_path:
            if self.train:
                img = Image.open(path + '/' + str(self.permutation_indoor[id]) + '.jpg')
            else:
                img = Image.open(path + '/' + str(self.permutation_indoor[id] + 10000) + '.jpg')
        else:
            if self.train:
                img = Image.open(path + '/' + str(self.permutation_outdoor[id]) + '.jpg')
            else:
                img = Image.open(path + '/' + str(self.permutation_outdoor[id] + 150) + '.jpg')

        img = img.convert('RGB') #Sometimes cv2.imread reads RGBA as RGB and after 'fro raw to requred there are still some JPEG'
        img_array = np.array(img)

        quot = min(img_array.shape[0], img_array.shape[1]) / 360
        if quot > 1:
            img = img.resize((int(img_array.shape[1] / quot), int(img_array.shape[0] / quot)))

        crop = ''
        if (is_reflection):
            crop = thv.transforms.RandomCrop(134)
        else:
            crop = thv.transforms.RandomCrop(128)
        croped_image = np.array(crop(img))

        if min(croped_image.shape[0], croped_image.shape[1]) < 128:
            raise Exception('image' + str(self.permutation_indoor[id]) + 'is too small')

        transposed_image = np.transpose(croped_image, (2, 0, 1))
        if (transposed_image.shape[0] == 4):
            raise Exception('image ' + str(self.permutation_indoor[id]) + 'is jpeg, but jpg is required')
        return th.Tensor(transposed_image / 255)
        #return th.Tensor(transposed_image)

    def __getitem__(self, id):
        indoor_item = self.__get_img(id, self.indoor_path, False)

        features = []
        labels = []
        for i in random.sample(range(self.outdoor_size), self.batch_size):
            outdoor_item = self.__get_img(i, self.outdoor_path, True) # shapes = [3, 128, 128]
            pare = self.mix_indoor_outdoor(indoor_item, outdoor_item)
            features.append(pare[0])
            labels.append(pare[1])
        # print(th.stack(features).shape, th.stack(labels).shape)
        return th.stack(features), th.stack(labels)

    def mix_indoor_outdoor(self, indoor_item, outdoor_item, double_reflec=True, blur=False):
        kernel_size = 7
        alpha = np.random.randint(75, 80 + 1) / 100
        pos_impulse1 = np.random.randint(0, 2 * (kernel_size - 1))
        pos_impulse2 = np.random.randint(0, 2 * (kernel_size - 1))

        kernel = np.zeros((kernel_size, kernel_size))
        alpha1 = 1 - np.sqrt(alpha)
        alpha2 = np.sqrt(alpha) - alpha

        x1 = 0 if pos_impulse1 < kernel_size - 1 else pos_impulse1 % (kernel_size - 1)
        y1 = pos_impulse1 % (kernel_size - 1) if pos_impulse1 < kernel_size - 1 else -1

        x2 = 0 if pos_impulse2 < kernel_size - 1 else pos_impulse2 % (kernel_size - 1)
        y2 = 1 + pos_impulse2 % (kernel_size - 1) if pos_impulse2 < kernel_size - 1 else -1

        kernel[x1][y1] = alpha1
        kernel[x2][y2] = alpha2
        kernel = np.repeat(kernel[None, None, :, :], 3, 0)

        if len(outdoor_item.shape) == 3:
            outdoor_item = outdoor_item[None, :, :, :]

        reflection = F.conv2d(outdoor_item, th.Tensor(kernel), padding=0, groups=3)
        #reflection = np.transpose(reflection.numpy().squeeze(), (1, 2, 0))

        """ prints three images
        #img = Image.fromarray(np.transpose(th.squeeze((1 - alpha1 - alpha2) * indoor_item[None, :, :, :] + reflection), (1, 2, 0)).numpy().astype('uint8'))
        #img.show()
        #(Image.fromarray(np.transpose(indoor_item.numpy(), (1, 2, 0)).astype('uint8'))).show()
        #(Image.fromarray(np.transpose(th.squeeze(outdoor_item).numpy(), (1, 2, 0)).astype('uint8'))).show()
        """

        mix = th.squeeze((1 - alpha1 - alpha2) * indoor_item[None, :, :, :] + reflection)
        row = th.Tensor(np.concatenate((indoor_item, th.squeeze(reflection)), axis=0))
        return mix, row

    def __iter__(self):
        return self

    def __next__(self):
        if self.id_indoor == self.indoor_size:
            self.id_indoor = 0
            raise StopIteration
        self.id_indoor += 1
        return self.__getitem__(self.permutation_indoor[self.id_indoor - 1])

#for i, (featurs, labels) in enumerate(MultiReflectionDataSet()):
#   print(featurs.shape, labels.shape)
'''
loader_indoors = ImageDataSet()
loader_indoors.__getitem__(4)
loader_outdoors = DataLoader(ImageDataSet('outdoor', n), batch_size=3)
for i, batch in enumerate(zip(loader_indoors, loader_outdoors)):
    pass
'''
