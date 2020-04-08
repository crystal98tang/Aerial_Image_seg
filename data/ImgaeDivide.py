import os
import random
from PIL import Image
import numpy as np
import imageio

kindOfImage = ["austin","chicago","kitsap","tyrol-w","vienna"]

def divide_dataset(data_path, train_dir, test_dir, kind_of_image):
    # box_list4train = init_box(1,750,5000,5000,256)
    num_each_label = 15
    box_list4test = init_box(0,0,5000,5000,256)
    # print("################Training Image################")
    # divide_train_image(data_path, train_dir, num_each_image, kind_of_image, box_list4train,mode="training")

    print("################Test Image################")
    divide_image(data_path, test_dir, num_each_label, box_list4test)


def divide_image(data_path, save_dir, num_big_image, box_list):
    cur_count = 0   # 可续---
    for i in range(0, num_big_image):
        img = np.array(imageio.imread(os.path.join(data_path, 'testing', "%02d.tif" % i)))
        valid = np.array(imageio.imread(os.path.join(data_path, 'testing_label', "%02d.tif" % i)))
        image_p = Image.fromarray(img)
        valid_p = Image.fromarray(valid)
        image_list = [np.array(image_p.crop(box)) for box in box_list]
        valid_list = [np.array(valid_p.crop(box)) for box in box_list]
        cur_count = save_patch_image(save_dir, image_list, valid_list, cur_count)
        print("finish:%d" % i)


def divide_train_image(data_path, save_dir, num_big_image, kinds, box_list, mode):
    cur_count = 0  # 可续---
    for i in range(0, num_big_image):
        for kind in kinds:
            if mode == 'training':
                img = np.array(imageio.imread(os.path.join(data_path, 'training', "%s%d.tif" % (kind, i))))
                valid = np.array(imageio.imread(os.path.join(data_path, 'validation', "%s%d.tif" % (kind, i))))
            elif mode == 'testing':
                img = np.array(imageio.imread(os.path.join(data_path, 'testing', "%s%d.tif" % (kind, i))))
                valid = np.array(
                    imageio.imread(os.path.join(data_path, 'testing_label', "%s%d.tif" % (kind, i))))
            else:
                raise Exception("mode Error")
            image_p = Image.fromarray(img)
            valid_p = Image.fromarray(valid)
            image_list = [np.array(image_p.crop(box)) for box in box_list]
            valid_list = [np.array(valid_p.crop(box)) for box in box_list]
            cur_count = save_patch_image(save_dir, image_list, valid_list, cur_count)


# Saving Image

def save_patch_image(save_dir, image_list, valid_list, c):
    #TODO：在patch_path保存两张patch
    image_path = os.path.join(save_dir, "image")
    valid_path = os.path.join(save_dir, "label")
    # dir exist
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    count = c
    # sum_pixel = image_list[0].shape[0] * image_list[0].shape[1]
    for image,annotation in zip(image_list,valid_list):
        # training use
        # if annotation.max() != 255:
        #     continue
        # tmp = np.count_nonzero(annotation)
        # if (tmp / sum_pixel) < 0.05:
        #     continue
        # if (tmp / sum_pixel) > 0.9:
        #     continue
        patch_image_file_path = os.path.join(image_path, str("%06d" % count)+ '.tif')
        patch_annotation_file_path = os.path.join(valid_path, str("%06d" % count)+ '.tif')

        imageio.imsave(patch_image_file_path, image)
        imageio.imsave(patch_annotation_file_path, annotation)
        count += 1

    return count

def init_box(mode, num, image_width, image_height, patch_size, overlay=0.875):
    """
    初始化剪切盒子
    :param mode: 0-顺序位移 1-随机位移
    :param num: 随机切割一张大图中切割的总数
    :param image_width: 宽
    :param image_height: 高
    :param patch_size: 切片尺寸
    :param overlay: 重复覆盖率
    """
    box_list = []
    # each pitch size
    item_width = patch_size  # patch size
    # count pitchs numbers
    cnt = 0
    # Use for Testing Big Image
    if mode == 0:
        # stride and iteration
        stride = item_width * (1 - overlay)
        x_iteration = int(image_width // stride) - int((patch_size - stride) // stride) - 1
        y_iteration = int(image_height // stride) - int((patch_size - stride) // stride) - 1

        for j in range(0, y_iteration):
            for i in range(0, x_iteration):
                cnt += 1
                # （left, upper, right, lower）
                box = (i * stride, j * stride, i * stride + patch_size, j * stride + patch_size)
                box_list.append(box)
        print("Big spilt sum:" + str(cnt))
        return box_list
    # Use for Training
    if mode == 1:
        # how many pitch to divide in one big image
        for i in range(0, num):
            cnt += 1
            x = random.randint(0, image_width - item_width)
            y = random.randint(0, image_height - item_width)
            # （left, upper, right, lower）
            box = (x, y, x + item_width, y + item_width)
            box_list.append(box)
        return box_list
    return -1

divide_dataset('../origin_Dataset', '../DataSet/IAILD/train', '../DataSet/IAILD/test_875', kindOfImage)
print("finish!")

