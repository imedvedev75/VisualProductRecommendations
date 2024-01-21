from PIL import Image
import os
import numpy as np


def read_image(image_file_path):
    if not os.path.exists(image_file_path):
        raise RuntimeError("File not found: {}".format(image_file_path))

    image_data = Image.open(image_file_path)

    # check image resolution fulfils maximum resolution requirement
    return scale_image(image_data, 299)


def scale_image(image_data, image_size):
    image_data = image_data.resize((image_size, image_size), resample=Image.BILINEAR)
    img_array = np.array(image_data)

    # handle grayscale images
    if len(img_array.shape) < 3:
        img_rgba = Image.new("RGBA", image_data.size)
        img_rgba.paste(image_data)
        img_array = np.array(img_rgba)

    #img_array = np.multiply(img_array, 1.0 / 127.5)
    #img_array = np.subtract(img_array, 1.0)
    img_array = img_array[:, :, :3]

    return img_array