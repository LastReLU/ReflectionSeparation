import os
import requests
from bs4 import BeautifulSoup

def get_img(id):
    url = 'https://www.hel-looks.com/big-photos/{}.jpg'.format(id)
    r = requests.get(url, allow_redirects=True)
    open('outdoor_imgs/img_{}.jpg'.format(id), 'wb').write(r.content)

def get_dataset():
    os.mkdir('outdoor_imgs')

    url = 'https://www.hel-looks.com/archive/#20190810_13'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', {'class': 'v'})
    for link in links:
        img_id = link.get('href')[1:]
        get_img(img_id)

get_dataset()