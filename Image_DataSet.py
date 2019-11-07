import os
import torchvision as thv
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch as th

class ImageDataSet(Dataset):
    def __init__(self, _len=1000):
        self.len = _len
        self.indoor_path = './data/indoor'
        self.outdoor_path = './data/indoor'
        self.permutation = np.random.permutation(self.len)

    def __len__(self):
        return self.len

    def __get_img(self, id, path):
        img = Image.open(path + '/' + str(self.permutation[id]) + '.jpg')
        img = img.convert('RGB') #Sometimes cv2.imread reads RGBA as RGB and after 'fro raw to requred there are still some JPEG'

        crop = thv.transforms.RandomCrop(128)
        croped_image = np.array(crop(img))

        if min(croped_image.shape[0], croped_image.shape[1]) < 128:
            raise Exception('image' + str(self.permutation[id]) + 'is too small')

        transposed_image = np.transpose(croped_image, (2, 0, 1))
        #print(transposed_image.shape)
        if (transposed_image.shape[0] == 4):
            raise Exception('image ' + str(self.permutation[id]) + 'is jpeg, but jpg is required')
        return th.Tensor((255 - transposed_image) / 255)

    def __getitem__(self, id):
        indoor_item = self.__get_img(id, self.indoor_path)
        outdoor_item = self.__get_img(id, self.outdoor_path)

        item = th.stack([indoor_item, outdoor_item])
        # print(item.shape)
        return item

'''
loader_indoors = ImageDataSet('indoor', 15305)
print(loader_indoors.__getitem__(4))
loader_outdoors = DataLoader(ImageDataSet('outdoor', n), batch_size=3)
for i, batch in enumerate(zip(loader_indoors, loader_outdoors)):
    pass
'''
