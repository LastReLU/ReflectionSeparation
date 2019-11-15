import cv2
import torch as th
import numpy as np

import sys
sys.path.append('../')

def export_img(img):
    return np.maximum((np.transpose(img.detach().numpy(), (1, 2, 0))*255).astype(int), 0)

def save_imgs(links):

    model = th.load('./app/model.hdf5', map_location=th.device('cpu'))
    model.eval()

    trans_links = []
    refl_links = []

    # print(type(url_names))

    # print(url_names)

    for link, name in links:
        trans_link = f'{link[:-4]}_trans.jpg'
        refl_link = f'{link[:-4]}_refl.jpg'
        trans_name = f'{name[:-4]}_trans.jpg'
        refl_name = f'{name[:-4]}_refl.jpg'

        img = cv2.imread(f'./uploads/{name}')
        img_float = img.astype(np.float32) / 255.0
        to_model_img = th.tensor(np.transpose(img_float, (2, 0, 1)))[None]

        from_model = model.predict(to_model_img)

        trans = from_model['img_trans'][0]
        refl = from_model['img_refl'][0]

        trans_links.append((trans_link, trans_name))
        refl_links.append((refl_link, refl_name))
        
        cv2.imwrite(f'./uploads/{trans_name}', export_img(trans))
        cv2.imwrite(f'./uploads/{refl_name}', export_img(refl))

    return (trans_links, refl_links)
