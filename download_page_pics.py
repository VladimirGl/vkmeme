from glob import glob
import requests
import shutil
import os

def get_pic(url, path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path + url.replace('/', '-'), 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

pic_files = glob('meme_pic_urls/*')

N = 0
done_file = open('done.txt', 'r')
done_pics = [line[:-1] for line in done_file]
done_file.close()

done_file = open('done.txt', 'a')

pic_files = list(set(pic_files) - set(done_pics))
pic_files = sorted(pic_files)

for pic_file in pic_files[100*N:(100*(N+1))]:
    path = 'meme_pages_pics/' + pic_file.split('/')[-1] + '/'

    try:
        os.mkdir(path)
    except:
        pass
    f = open(pic_file, 'r')
    print(pic_file)

    for pic in f:
        if 'http' not in pic:
            pic = 'http:' + pic

        if len(pic) < 10:
            continue
        if 'gravatar' in pic or 'avatars' in pic or 'userapi' in pic or 'boombox_logo' in pic:
            continue
        else:
            get_pic(pic[:-1], path)

    done_file.write(pic_file + '\n')