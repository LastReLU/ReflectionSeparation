import cv2
import os
import numpy as np

def replace(file_name, dir_name):
    os.replace('{}/{}'.format(dir_name, file_name), '{}_grey/{}'.format(dir_name, file_name))

def main():
    dir_name = input()
    list_of_img = os.listdir(dir_name)
    if not len(list_of_img):
        exit()
    
    os.mkdir('{}_grey'.format(dir_name))

    for img_name in list_of_img:
        img = cv2.imread('{}/{}'.format(dir_name, img_name), cv2.COLOR_BGR2RGB)
        img_shape = len(img.shape)
        if img_shape < 3:
            replace(img_name, dir_name)
        else:
            img = np.transpose(img, (2, 0, 1))
            if (((img[0] == img[1]) & (img[1] == img[2])) == False).sum() == 0:
                replace(img_name, dir_name)

    os.rename(dir_name, '{}_rgb'.format(dir_name))

main()
