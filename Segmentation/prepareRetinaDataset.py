from PIL import Image
import cv2
import random
import numpy as np
import os
import glob

FOLDER = 'training/'
DIR = 'D:/Alexey/Projects/retina-unet/DRIVE/' + FOLDER
DIR_OUT = 'D:/Alexey/Projects/data/DRIVE/DRIVE_128/' + FOLDER
SIZE = 128

def createDataset():
    images = os.listdir(DIR + '/images')

    for im_file in images:

        im = Image.open(DIR + 'images/' + im_file)
        im_mask = Image.open(DIR + 'mask/' + im_file[0:-4] + '_mask.gif')
        im_label = Image.open(DIR + '1st_manual/' + im_file[0:3] + 'manual1.gif')

        w,h = im.size
        count = 0

        while True:
            x = random.randint(0, w - SIZE)
            y = random.randint(0, h - SIZE)

            if np.all(np.array(im_mask.crop((x,y,x+SIZE,y+SIZE)), np.bool) == True):
                count+=1
                im_patch = im.crop((x, y, x + SIZE, y + SIZE))
                im_patch_label = im_label.crop((x, y, x + SIZE, y + SIZE))
                im_patch.save(DIR_OUT + im_file[0:3] + 'patch' + str(count) + '.jpg')
                im_patch_label.save(DIR_OUT + im_file[0:3] + 'patch' + str(count) + '_mask.png')
                if count == 100:
                    break


def countClassBalance():
    files = glob.glob(DIR_OUT + '*_mask.tif')

    countAll = 0
    countWhite = 0

    for file in files:
        im = np.array(Image.open(file), np.byte)
        countAll += im.shape[0] * im.shape[1]
        countWhite += np.sum(im == -1)
    print('all: %s, black: %s, white: %s' % (countAll, countAll - countWhite, countWhite))


#countClassBalance()
#im=cv2.imread('C:/Users/d038471/Google Drive/Projects/commerceapp/data/DRIVE_50/training/21_patch1_mask.tif')
#cv2.imshow('om',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

createDataset()