import os
import random
import numpy as np
import imageio
import matplotlib.pyplot as plt
""""""
# If can not work, add "tensorflow.python."
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import to_categorical
#
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical
#
""""""
seed = random.randint(1,10001)

def adjustVaild(mask):
    if max(mask):
        return 1
    return 0

# label information of 5 classes:
# background
# RGB:(0,0,0) 黑 0
########################
# water
# RGB:(0,0,1) 蓝 1
# farmland
# RGB:(0,1,0) 绿 2
# forest
# RGB:(0,1,1) 淡蓝 3
# built-up
# RGB:(1,0,0) 红 4
# meadow
# RGB:(1,1,0) 黄 5

KindOfName = ['water', 'farmland', 'forest', 'built-up ', 'meadow']
# FIXME:适配多分类 把多色转单色
def adjust_data(img, mask, classes):
    """
    :param img:
    :param mask:
    :return:
    """
    img = img / 255
    if classes == 2:
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    else:
        #TODO: 尝试转换单色
        return 1

    mask = to_categorical(mask, num_classes=classes)
    return img,mask

# 数据增广 处理后
def show(img, mask):
    k = 0
    for i in img:
        imageio.imwrite(os.path.join("test_img/", "%d_img.jpg" % k), i.astype(np.uint8))
        k += 1
    k = 0
    for i in mask:
        imageio.imwrite(os.path.join("test_mask/", "%d_mask.jpg" % k), i.astype(np.uint8))
        k += 1