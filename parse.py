from bs4 import BeautifulSoup
import pandas as pd
from glob import glob

def get_meme_info(article):
    a = article.find('a', href=True)
    url = a['href']
    title = a['title']
    pic_url = a.div.img['src']

    n_views = article.find("span", {"class": "count"}).text

    return title, url, pic_url, n_views

result = open('result.csv', 'a')

all_pages = glob('memepedia_main/*')

fname = 'index.html.2'

for page in all_pages:
    print(page)
    with open(page) as f:
        soup = BeautifulSoup(f, features="html.parser")

        articles = soup.find_all('article')

        for i, article in enumerate(articles):
            try:
                meme_info = get_meme_info(article)
                result.write('|||'.join(meme_info) + '\n')
            except:
                pass


result.close()