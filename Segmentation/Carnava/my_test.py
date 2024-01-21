import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import params
import os

import glob

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model_factory()

DIR = 'D:/Alexey/Projects/data/sh_hv'

#df_train = pd.read_csv(DIR + '/train_masks.csv')
#ids_train = df_train['img'].map(lambda s: s.split('.')[0])
#ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.001, random_state=42)

def get_ids_train_drive():
    ids_train = []
    for file in os.listdir(DIR + '/test'):
        l = file.split('.')
        if l[1]=='jpg' and 'mask' not in l[0]:
            ids_train.append(l[0])
    return np.array(ids_train)

ids_train = get_ids_train_drive()
ids_valid_split=ids_train[:50]


x_batch = []
y_batch = []
for id in ids_valid_split:
    img = cv2.imread(DIR + '/test/{}.jpg'.format(id))
    img = cv2.resize(img, (input_size, input_size))
    img_mask = cv2.imread(DIR + '/test/{}_mask.jpg'.format(id))
    img_mask = cv2.resize(img_mask, (input_size, input_size))
    x_batch.append(img)
    y_batch.append(img_mask)

x_batch = np.array(x_batch, np.float32) / 255
y_batch = np.array(y_batch, np.float32) / 255

model.load_weights(filepath=DIR + '/weights/best_weights.hdf5')
preds = model.predict_on_batch(x_batch)
#preds = np.squeeze(preds, axis=3)

pred_batch = []
for pred in preds:
    #prob = cv2.resize(pred, (orig_width, orig_height))
    prob = pred
    mask = prob > threshold
    pred_batch.append(mask)

pred_batch = np.squeeze(np.array(pred_batch, np.bool), axis=3)

pred_imgs = np.zeros((pred_batch.shape[0], pred_batch.shape[1], pred_batch.shape[2], 3), dtype=np.byte)
pred_imgs[..., 0] = pred_batch
pred_imgs[..., 1] = pred_batch
pred_imgs[..., 2] = pred_batch

im = np.uint8(np.concatenate((x_batch * 255, y_batch * 255, pred_imgs * 255), axis=2))
im = im.reshape(-1, im.shape[2], im.shape[3])
#im = Image.fromarray(im)
cv2.imwrite(DIR + '/test_results.jpg', im)