import sys
import os
from PIL import Image
import numpy as np


sys.path.append(os.getcwd() + '/tf_unet')

from tf_unet import unet, util, image_util

#output_path = 'unet_out_car'
output_path = '/nfs/users/alexey/unet_out_car'

#DATA_PATH = "D:/Alexey/Projects/data/Carnava/train_128/*.jpg"
#DATA_PATH = '/nfs/users/alexey/DRIVE_50/training/*.tif'
#DATA_PATH = 'car_train_128/*.jpg'
DATA_PATH = '/nfs/users/alexey/Car/train_128/*.jpg'

def train():
    #preparing data loading
    data_provider = image_util.ImageDataProvider(DATA_PATH, data_suffix=".jpg", mask_suffix='_mask.gif')

    #setup & training
    net = unet.Unet(layers=3, features_root=64, channels=3, n_class=2, cost="cross_entropy"
                    #,cost_kwargs={"class_weights":[0.1379, 0.8621]
                        #,"regularizer":0.00001}
                    )
    trainer = unet.Trainer(net, batch_size=30, optimizer='momentum')
    path = trainer.train(data_provider, output_path, training_iters=20, epochs=100)

#verification
def predict():
    file = 'C:/Users/d038471/Google Drive/Projects/commerceapp/data/DRIVE_50/test/01_patch1.tif'
    file_label = 'C:/Users/d038471/Google Drive/Projects/commerceapp/data/DRIVE_50/test/01_patch1_mask.tif'

    data = [Image.open(file)]
    label = [np.array(Image.open(file_label), np.bool)]

    net = unet.Unet(layers=3, features_root=64, channels=3, n_class=2)
    prediction = net.predict(output_path, data)

    unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))

    img = util.combine_img_prediction(data, label, prediction)
    #util.save_image(img, "prediction.jpg")


#predict()
train()