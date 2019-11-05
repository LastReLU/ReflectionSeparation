# This skript makes another folder, with images names form 0 to 15620, there you run it
import os
from shutil import copyfile
import numpy as np
import cv2
from PIL import Image


def check_RGB(img):
    if len(img.shape) == 2: # check that image is RGB
        return False
    x = np.transpose(img, (2, 0, 1))
    if x.shape[0] != 3: # check that image is jpg, not jpeg
        return False
    if min(x.shape[1], x.shape[2]) < 128: #check that image is large enought
        return False
    return (((x[0] == x[1]) & (x[1] == x[2])) == False).sum() != 0


id_ = 0
os.mkdir(os.getcwd() + '/mapped_ids')
for current_dir, dirs, files in os.walk(os.getcwd() + '/indoorCVPR_09'):
    for file in files:
        cur_name = str(id_) + '.jpg'
        src = current_dir + '/' + file

        #img = cv2.imread(src) doesn't work with jpeg files
        img = np.array(Image.open(src))

        if check_RGB(img):
            copyfile(src, os.getcwd() + '/mapped_ids/' + cur_name)
            id_ += 1

print(15620 - id_)
