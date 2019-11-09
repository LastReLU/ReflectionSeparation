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
    if min(x.shape[1], x.shape[2]) < 134: #check that image is large enought
        return False
    return (((x[0] == x[1]) & (x[1] == x[2])) == False).sum() != 0

def chang_data(path):
    amount_before = 0
    id_ = 0
    newpath = os.getcwd() + '/' + path
    os.mkdir(path)
    path = newpath + '_row'
    for current_dir, _, files in os.walk(path):
        for file in files:
            cur_name = str(id_) + '.jpg'
            src = current_dir + '/' + file

            img = cv2.imread(src, cv2.COLOR_BGR2RGB)
            if check_RGB(img):
                copyfile(src, newpath + '/' + cur_name)
                id_ += 1
            os.remove(src)
            amount_before += 1
    os.rmdir(path)
    return (id_, amount_before)

pare = chang_data('data/indoor')
pare2 = chang_data('data/outdoor')
print('There are ' + str(pare[0]) + ' sutable files of ' + str(pare[1]) + ' from indoor')
print('Same about outdoor', pare2[0], pare2[1])