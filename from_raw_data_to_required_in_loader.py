# This skript makes another folder, with images names form 0 to 15620, there you run it
import os
from shutil import copyfile
import numpy as np
import cv2
from PIL import Image
import sys

def check_RGB(img):
    if img is None: # check that image is JPEG
        return False
    if len(img.shape) == 2: # check that image is RGB
        return False
    x = np.transpose(img, (2, 0, 1))
    if min(x.shape[1], x.shape[2]) < 128: #check that image is large enought
        return False
    return (((x[0] == x[1]) & (x[1] == x[2])) == False).sum() != 0

if len(sys.argv) != 2:
    print('Enter distination folder')
else:
    id_ = 0
    path = os.getcwd() + sys.argv[1]
    for current_dir, _, files in os.walk(path):
        for file in files:
            cur_name = str(id_) + '.jpg'
            src = current_dir + '/' + file

            img = cv2.imread(src, cv2.COLOR_BGR2RGB)
            # img = np.array(Image.open(src))

            if check_RGB(img):
                # copyfile(src, os.getcwd() + '/mapped_ids/' + cur_name)
                os.rename(src, path + cur_name)
                id_ += 1
    print(15620 - id_)