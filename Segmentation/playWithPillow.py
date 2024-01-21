from PIL import Image
import cv2
import numpy as np
import os

DIR_IN = "D:/Alexey/Projects/data/Carnava/train_masks/"
DIR_OUT = "D:/Alexey/Projects/data/Carnava/train_masks_png/"

for file in os.listdir(DIR_IN):
    im = Image.open(DIR_IN + file)
    filename, file_extension = os.path.splitext(file)
    im.save(DIR_OUT + filename + '.png')
