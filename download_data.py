import os
import tarfile
import requests
from bs4 import BeautifulSoup

def get_outdoor_img(id, dir_name):
    url = 'https://www.hel-looks.com/big-photos/{}.jpg'.format(id)
    r = requests.get(url, allow_redirects=True)
    open('data/{}/{}.jpg'.format(dir_name, id), 'wb').write(r.content)


def replace(file_name, from_dir, to_dir):
    os.replace('{}/{}'.format(from_dir, file_name),
               '{}/{}'.format(to_dir, file_name))

def outdoor(dir_name):
    os.mkdir('data/{}'.format(dir_name))

    url = 'https://www.hel-looks.com/archive/#20190810_13'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', {'class': 'v'})
    for link in links:
        img_id = link.get('href')[1:]
        get_outdoor_img(img_id, dir_name)

def indoor(dir_name):
    os.mkdir('data/{}'.format(dir_name))

    # url = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
    # r = requests.get(url, allow_redirects=True)
    # open('data/indoor_img.tar', 'wb').write(r.content)
    
    tar =  tarfile.open('indoorCVPR_09.tar', 'r:')
    tar.extractall('data/indoor_img')
    tar.close()

    for current_dir, dirs, files in os.walk('data/indoor_img'):
        for file in files:
            replace(file, current_dir, 'data/{}'.format(dir_name))
        if current_dir != 'data/indoor_img' and current_dir != 'data/indoor_img/Images':
            os.rmdir(current_dir)

    os.rmdir('data/indoor_img/Images')
    os.rmdir('data/indoor_img')


def init():
    os.mkdir('data')
    indoor('indoor_row')
    outdoor('outdoor_row')

init()
