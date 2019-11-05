import os
import requests
from bs4 import BeautifulSoup

def get_img(id, img_num):
    url = 'https://www.hel-looks.com/big-photos/{}.jpg'.format(id)
    r = requests.get(url, allow_redirects=True)
    open('outdoor/{}.jpg'.format(img_num), 'wb').write(r.content)

def get_dataset():
    os.mkdir('outdoor')

    url = 'https://www.hel-looks.com/archive/#20190810_13'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', {'class': 'v'})
    for link, img_num in zip(links, range(len(links))):
        img_id = link.get('href')[1:]
        get_img(img_id, img_num)

get_dataset()