# -*- coding: cp852 -*-

import urllib.request as req
import re
import os
import unicodedata

DATADIR = 'd:/Alexey/Projects/data/background/plants'

if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

#response = req.urlopen(request)

#html = response.read()
#html = str(html)
#ucontent = html.decode('unicode_escape')

html = open('input.txt', encoding='utf-8').read().replace('\n', '')

pImg = re.compile('"ou":"http.*?jpg')

l = pImg.findall(html)

for imgtag in l:
    imgurl = imgtag.replace('"ou":"', '').replace('"', '')
    name = imgtag.split('/')[-1]
    #name = name + '.jpg'
    try:
        opener = req.build_opener()
        opener.addheaders = [
            ('User-agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0')]
        req.install_opener(opener)
        req.urlretrieve(imgurl, DATADIR + '/' + name)
    except:
        pass

