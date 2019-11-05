import os
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset

class ImageDataSet(Dataset):
    def __init__(self, class_='indoors', len_=15620 - 185):  # 185 is amount of jpeg, black and wite, and too small
        # picters, 15620 is a size of indoors dataset
        self.len = len_
        self.path = ''
        if class_ == 'indoors':
            self.path = os.getcwd() + '/indoor'
        elif class_ == 'outdoors':
            self.path = os.getcwd() + '/outdoor'
        else:
            raise Exception()
        self.permutation = np.random.permutation(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, id):
        img = Image.open(self.path + '/' + str(self.permutation[id]) + '.jpg')
        # img = img.convert('RGB')

        # img = cv2.imread(self.path + '/' + str(self.permutation[id]) + '.jpg', cv2.COLOR_BGR2RGB)

        crop = torchvision.transforms.RandomCrop(128)
        croped_image = np.array(crop(img))

        if min(croped_image.shape[0], croped_image.shape[1]) < 128:
            raise Exception('image' + str(self.permutation[id]) + 'is too small')

        transposed_image = np.transpose(croped_image, (2, 0, 1))
        if (transposed_image.shape[0] == 4):
            raise Exception('image ' + str(self.permutation[id]) + 'is jpeg, but jpg is required')

        return transposed_image

imageDataSet = ImageDataSet('indoors')
loader = DataLoader(imageDataSet, batch_size=3)
for i in loader:
    pass
