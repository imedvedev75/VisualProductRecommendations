import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from scipy import misc
from PIL import Image

# random example images
#images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
image = misc.imread('../data/sampleShoes/shoe1.jpg')
image = image[:, :, :3]
print(image.shape)
images = np.array([image])

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

seq = iaa.Invert(1.0, per_channel=True)
seq = iaa.Affine(shear=(-40, 40))
seq = iaa.Affine(shear=(-40, 40))

images_aug = seq.augment_images(images)
images_aug = Image.fromarray(images_aug[0], 'RGB')
images_aug.show()
#print(images_aug.shape)
#seq.show_grid(images[0], cols=8, rows=8)


