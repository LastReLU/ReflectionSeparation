import os
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

class ImageDataSet(Dataset):
    def __init__(self, class_='indoors', len_=15305):
        self.len = len_
        self.path = ''
        if class_ == 'indoor':
            self.path = os.getcwd() + '/data/indoor'
        elif class_ == 'outdoor':
            self.path = os.getcwd() + '/data/outdoor'
        else:
            raise Exception()
        self.permutation = np.random.permutation(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, id):
        img = Image.open(self.path + '/' + str(self.permutation[id]) + '.jpg')
        img = img.convert('RGB') #Sometimes cv2.imread reads RGBA as RGB and after 'fro raw to requred there are still some JPEG'

        crop = torchvision.transforms.RandomCrop(128)
        croped_image = np.array(crop(img))

        if min(croped_image.shape[0], croped_image.shape[1]) < 128:
            raise Exception('image' + str(self.permutation[id]) + 'is too small')

        transposed_image = np.expand_dims(np.transpose(croped_image, (2, 0, 1)), axis=0)
        #print(transposed_image.shape)
        if (transposed_image.shape[0] == 4):
            raise Exception('image ' + str(self.permutation[id]) + 'is jpeg, but jpg is required')

        return transposed_image

'''
loader_indoors = ImageDataSet('indoor', 15305)
loader_indoors.__getitem__(4)
loader_outdoors = DataLoader(ImageDataSet('outdoor', n), batch_size=3)
for i, batch in enumerate(zip(loader_indoors, loader_outdoors)):
    pass
'''