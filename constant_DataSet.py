import os
import torch as th
import cv2
import numpy as np

class const_DataSet():
    def __init__(self, test=False):
        path = os.getcwd() + '/data'
        self.test = test
        self.shift = 0
        if test:
            self.shift += 4000 * 18
        self.path_image = path + '/images'
        self.path_reflection = path + '/reflection'
        self.path_transmission = path + '/transmission'

    def __len__(self):
        return 5 * 8 - self.shift

    def __getitem__(self, id):
        if self.test:
            id += self.shift
        image = np.transpose(cv2.imread(self.path_image + '/' + str(id) + '.jpg'), (2, 0, 1)) # BGR !!!
        transmission = np.transpose(cv2.imread(self.path_transmission + '/' + str(id) + '.jpg'), (2, 0, 1))
        # print(image.shape, transmission.shape)
        return th.Tensor(np.concatenate((image / 255, transmission / 255), 0))

# cds = const_DataSet(False)
# check = cds[1]
# print(check.shape)
