# -*- coding: cp852 -*-

import urllib.request as req
import re
import os

DATADIR = 'D:/Alexey/Projects/data/sh_2'

url = "https://www.amazon.de/s/ref=s9_acss_bw_cg_WSHC_3d1_w?__mk_de_DE=%C5M%C5Z%D5%D1&rh=i%3Ashoes%2Cn%3A7003984031%2Cn%3A1760313031&bbn=7003984031&ie=UTF8&lo=shoes&pf_rd_m=A3JWKAKR8XB7XF&pf_rd_s=merchandised-search-6&pf_rd_r=2QXCABCQA9JKEKXJZYA1&pf_rd_t=101&pf_rd_p=ee29d923-efa2-43c1-b8ce-0ac39d0c5b3f&pf_rd_i=1760304031"
category = 'women schnuerhalbschuhe'
keywords = ['schnuerhalbschuhe']

def replaceName(name):
    name = name.replace('"', '').replace('alt=', '').replace('&#246;', 'oe').replace('&#223;', 'ss'). \
        replace('&#228;', 'ae').replace('&#252;', 'ue').replace('&#220;', 'Ue'). \
        replace('&#214;', 'Oe').replace('&#196;', 'Ae').replace('&reg;', ''). \
        replace('/', '').replace('&#8211;', ' - ').replace('|', '').replace('*','').\
        replace(';',' ').replace(':', '')
    return name[:100]


response = req.urlopen(url)
html = response.read()
html = html.decode('utf-8').replace('\n', '')

"""
pCategory = re.compile('<title>.*?</title>')
category = pCategory.findall(html)[0]
category = category.replace('Amazon.de:', '').replace('<title>','').replace('</title>','').replace('&#039;', '').split(':')[0].strip()
"""

pathToStore = DATADIR + '/' + category
if not os.path.exists(pathToStore):
    os.makedirs(pathToStore)


for i in range(0,50): #10 pages
    allowed_symbols = '[a-zA-Z0-9.,_-]'
    pImg = re.compile('<img src="https://images-eu.ssl-images-amazon.com/images/I/' + allowed_symbols + '*200,260_.jpg".*?alt=".*?".*?>')
    pUrl = re.compile('"https://images-eu.ssl-images-amazon.com/images/I/' + allowed_symbols + '*200,260_.jpg"')
    pText = re.compile('alt=".*?"')
    #p2 = re.compile(allowed_symbols + '*200,260_.jpg')

    l = pImg.findall(html)

    for imgtag in l:
        imgurl = pUrl.findall(imgtag)[0]
        imgurl = imgurl[1:len(imgurl)-1]
        name = pText.findall(imgtag)[0]
        name = replaceName(name) + '.jpg'
        if any(ext in name.lower() for ext in keywords):
            try:
                req.urlretrieve(imgurl, pathToStore + '/' + name)
            except:
                print('error retrieving image: ' + name)

    """
    pNextPage = re.compile('<a title="N.*?chste Seite".*?</a>')
    nextPage = pNextPage.findall(html)
    nextPage = nextPage[0]

    pNextPageUrl = re.compile('href=".*?"')
    nextPageUrl = pNextPageUrl.findall(nextPage)[0]
    nextPageUrl = nextPageUrl.replace('href="', '').replace('"', '')
    if nextPageUrl.find('https://www.amazon.de') == -1:
        nextPageUrl = 'https://www.amazon.de' + nextPageUrl
    """

    nextPageUrl = url + '&page=' + str(i + 2)

    print('retrieving url for page: ' + str(i+2))

    while True:
        try:
            response = req.urlopen(nextPageUrl)
            html = response.read().decode('utf-8').replace('\n', '')
            break
        except:
            print('error retrieving url')


