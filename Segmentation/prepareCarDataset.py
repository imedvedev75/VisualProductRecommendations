from PIL import Image
import cv2
import numpy as np
import os

DIR = "D:/Alexey/Projects/data/Carnava/train/"
DIR_MASK = "D:/Alexey/Projects/data/Carnava/train_masks_gif/"
DIR_OUT = "D:/Alexey/Projects/data/Carnava/train_128/"

"""
for file in os.listdir(DIR):
    im = cv2.imread(DIR + file)
    im = cv2.resize(im, (128, 128))
    cv2.imwrite(DIR_OUT + file, im)
    #im = Image.open(DIR + file)
    #im.thumbnail((128,128))
    #im.save(DIR_OUT + file)
"""

for file in os.listdir(DIR_MASK):
    #im = cv2.imread(DIR_MASK + file)
    #im = cv2.resize(im, (128, 128))
    #cv2.imwrite(DIR_OUT + file, im)
    im = Image.open(DIR_MASK + file)
    im = im.resize((128,128))
    im.save(DIR_OUT + file)