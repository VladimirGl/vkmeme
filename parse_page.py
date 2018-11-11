from bs4 import BeautifulSoup
import pandas as pd
from glob import glob

def get_meme_info(article):
    a = article.find("div", {"class": "bb-row"})

    title = article.find("meta", {"property": "og:title"})['content']
    description = article.find("meta", {"property": "og:description"})['content']

#    a = article.find("div", {"class": "site-main"})
    tt = article.findAll('p')

    imgs_t = []
    for x in tt:
        try:
            imgs_t.append(x.img['src'])
        except:
            pass

    imgs = a.findAll('img')
    imgs = imgs_t + [img['src'] for img in imgs]

    try:
        pics_2 = article.findAll("figure", {"class": "small-gallery-image"})
        imgs_2 = [img.img['src'] for img in pics_2]

        imgs = imgs + imgs_2
    except:
        pass

    return title, description, list(set(imgs))

result = open('result_pages.csv', 'a')

all_pages = glob('meme_pages/*')

for page in all_pages:
    print(page)
    with open(page) as f:
        soup = BeautifulSoup(f, features="html.parser")

        try:
            title, descr, images = get_meme_info(soup)
            result.write('|||'.join([page,title, descr]) + '\n')
            new_pics = 'meme_pic_urls/{}.txt'.format(page.split('/')[-1])
            pic_file = open(new_pics, 'w')
            for img in images:
                pic_file.write(img + '\n')
            pic_file.close()
        except:
            pass


result.close()