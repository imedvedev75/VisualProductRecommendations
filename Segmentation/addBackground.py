from PIL import Image
import numpy as np

DIR = 'C:/Users/d038471/Google Drive/Projects/commerceapp/data/shoes_2/women trainers'

im = Image.open(DIR + '/' + '20% Rabatt ab 2 Paare! JOOMRA Unisex leicht Mesh atmungsaktiv laufschuhe 36-46.jpg')
im_back = np.array(Image.open('D:/Alexey/Projects/data/lm/lam_aug/0000 Extravagant Dynamic Laminat Landhausdiele S Eiche Alpin/Image1_299.jpg'))

im_arr = np.array(im, np.uint8)
im_mask = np.zeros((im_arr.shape[0],im_arr.shape[1],3), np.uint8)

for i in range(im_arr.shape[0]):
    for j in range(im_arr.shape[1]):
        if np.all(im_arr[i,j] == 255):
            im_arr[i, j] = im_back[i,j]
            im_mask[i,j] = [0,0,0]
        else:
            im_mask[i, j] = [255, 255, 255]

im_aug = Image.fromarray(im_arr)
im_mask = Image.fromarray(im_mask)

im_aug.show()
im_mask.show()
