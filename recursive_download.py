import os
from pathlib import Path
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

def get_links(content):
    soup = BeautifulSoup(content)
    for a in soup.findAll('a'):
        yield a.get('href')

def download(url):
    path = urlparse(url).path.lstrip('/')
    print(path)
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception('status code is {} for {}'.format(r.status_code, url))
    content = r.text
    if path.endswith('/'):
        Path(path.rstrip('/')).mkdir(parents=True, exist_ok=True)
        for link in get_links(content):
            if not link.startswith('.'): # skip hidden files such as .DS_Store
                download(urljoin(url, link))
    else:
        with open(path, 'w') as f:
            f.write(content)


if __name__ == '__main__':
    # the trailing / indicates a folder
    url = 'http://ed470d37.ngrok.io/exp/network/'
    download(url)
