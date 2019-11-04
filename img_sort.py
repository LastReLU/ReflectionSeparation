import cv2
import os

list_of_img = os.listdir("indoor_imgs")

if not len(list_of_img):
    exit()

os.mkdir('indoor_gray')

for img_name in list_of_img:
    img_array = cv2.imread('indoor_imgs/' + img_name, cv2.COLOR_BGR2RGB)
    img_shape = len(img_array.shape)
    print(img_array.shape)
    if img_shape < 3:
        os.replace('indoor_imgs/' + img_name, 'indoor_gray/' + img_name)

os.rename('indoor_imgs', 'indoor_rgb')