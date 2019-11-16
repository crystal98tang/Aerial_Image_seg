import os
import random
import numpy as np
import imageio
import scipy.misc as misc
# If can not work, add "tensorflow.python."
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
#
seed = random.randint(1,10001)
#
def adjustData(img,mask):
    """
    :param img:
    :param mask:
    :return:
    """
    if np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    mask = to_categorical(mask, num_classes=2)
    return img,mask