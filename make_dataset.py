import os
import numpy as np
from PIL import Image
import torchvision as thv
import torch as th
import torch.nn.functional as F
import cv2

MAX_OUTDOOR = 2700
MAX_INDOOR = 5000

cur_path = os.getcwd() + '/data' # path to current directory
print(cur_path)
try:
    os.makedirs(cur_path + '/images')
    os.makedirs(cur_path + '/transmission')
    os.makedirs(cur_path + '/reflection')
except:
    print('You already have folders that this skript creates, delete them and run this skript again if it makes sense')

arange = np.arange(MAX_OUTDOOR)
blur_kernel_size = np.array([1, 3, 5])

def resize_and_crop(image, crop_size):
    img_array = np.array(image)
    quot = min(img_array.shape[0], img_array.shape[1]) / 360
    if quot > 1:
        image = image.resize(
            (int(img_array.shape[1] / quot), int(img_array.shape[0] / quot)))
    # random.seed(seed)
    crop = thv.transforms.RandomCrop(crop_size)
    return crop(image)

def add_blur(image, blur_kernel_size):
    return cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

def add_double_reflection(image, alpha, first_ref_pos, second_ref_pos):
    kernel_size = 7
    kernel = np.zeros((kernel_size, kernel_size))
    alpha1 = 1 - np.sqrt(alpha)
    alpha2 = np.sqrt(alpha) - alpha

    (x1, y1) = first_ref_pos
    (x2, y2) = second_ref_pos

    kernel[x1, y1] = alpha1
    kernel[x2, y2] = alpha2
    kernel = np.repeat(kernel[None, None, :, :], 3, 0)

    if len(image.shape) == 3:
        image = image[None, :, :, :]
    if image.shape[3] == 3:
        image = th.Tensor(np.transpose(image, (0, 3, 1, 2)))
    if isinstance(image, np.ndarray):
        image = th.Tensor(image)

    reflection = F.conv2d(image, th.Tensor(kernel), groups=3)
    reflection = reflection.numpy().squeeze()

    return reflection

cnt = 0
for cur_dir, dirs, files in os.walk(cur_path + '/indoor'):
    ids = np.random.choice(arange, 18)
    for file in files:
        if cnt == MAX_INDOOR:
            break

        transmissiom_image = Image.open(cur_path + '/indoor/' + file)
        transmissiom_image = transmissiom_image.convert('RGB')
        croped_transmission = resize_and_crop(transmissiom_image, 128)
        cnt_cur = 0
        for i in ids:
            reflection_image = Image.open(cur_path + '/outdoor/' + str(i) + '.jpg')
            reflection_image = reflection_image.convert('RGB')
            #to RGB
            croped_reflection = resize_and_crop(reflection_image, 134)
            blured_croped_reflection = add_blur(np.transpose(np.array(croped_reflection), (2, 0, 1)), np.random.choice(blur_kernel_size, 1))

            alpha = np.random.uniform(0.75, 0.8)
            pos = (np.random.randint(0, 7), np.random.randint(0, 7))
            pos2 = (np.random.randint(0, 7), np.random.randint(0, 7))

            final_reflection = np.transpose(add_double_reflection(blured_croped_reflection, alpha, pos, pos2), (1, 2, 0))
            final_transmission = alpha * np.array(croped_transmission)
            final_image = final_transmission + final_reflection

            #print(cur_path + '/transmission/' + str(file))
            cv2.imwrite(cur_path + '/transmission/' + str(18 * cnt + cnt_cur) + '.jpg', final_transmission.astype('uint8'))
            cv2.imwrite(cur_path + '/reflection/' + str(18 * cnt + cnt_cur) + '.jpg', final_reflection.astype('uint8'))
            cv2.imwrite(cur_path + '/images/' + str(18 * cnt + cnt_cur) + '.jpg', final_image.astype('uint8'))

            cnt_cur += 1
            #Image.fromarray(final_reflection.astype('uint8')).save(cur_path + '/reflection/' + str(file))
            #Image.fromarray(final_image.astype('uint8')).save(cur_path + '/images/' + str(file))

        cnt += 1