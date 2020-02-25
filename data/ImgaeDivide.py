import os
import random
import numpy as np
import imageio

KindOfName = ['water', 'farmland', 'forest', 'built-up ', 'meadow']
KindOfImage = [[], [], [], [], []]


def read_info(list_file_path):
    i = 0
    for name in KindOfName:
        with open(os.path.join(list_file_path, "%s.txt" % name), "r") as f:
            KindOfImage[i].append(f.readlines())
            i += 1


def divide_dataset(data_path, train_dir, test_dir, num_each_image, patch_size):
    """
    切割主入口
    :param data_path:
    :param train_dir:
    :param test_dir:
    :param num_each_image:
    :param patch_size:
    :return:
    """
    #TODO:自动读取原始图片尺寸
    w = 5000
    h = 5000
    #EndTODO
    box_list4train = init_box(1, w, h, patch_size)
    box_list4test = init_box(0, w, h, patch_size)
    #FIXME:修改后更新divide_image调用参数
    print("################Training Image################")
    divide_image(data_path, train_dir, num_each_image, kind_of_image, box_list4train, mode="training")
    print("################Test Image################")
    divide_image(data_path, test_dir, 3, kind_of_image, box_list4test, mode="testing")
    #EndFIXME


#FIXME:适配多数据集
def divide_image(data_path, save_dir, num_big_image, kinds, box_list,mode):
    """
    切割任务具体实现
    :param data_path:
    :param save_dir:
    :param num_big_image:
    :param kinds:
    :param box_list:
    :param mode:
    :return:
    """
    cur_count = 0
    for i in range(0, num_big_image):
        for kind in kinds:
            if mode == 'training':
                img = np.array(imageio.imread(os.path.join(data_path,'training',"%s%d.tif"%(kind,i))))
                valid = np.array(imageio.imread(os.path.join(data_path,'validation',"%s%d.tif"%(kind,i))))
            elif mode == 'testing':
                img = np.array(imageio.imread(os.path.join(data_path, 'testing', "%s%d.tif" % (kind, i))))
                valid = np.array(imageio.imread(os.path.join(data_path, 'testing_label', "%s%d.tif" % (kind, i))))
            else:
                raise Exception("mode Error")

            image_p = Image.fromarray(img)
            valid_p = Image.fromarray(valid)
            image_list = [np.array(image_p.crop(box)) for box in box_list]
            valid_list = [np.array(valid_p.crop(box)) for box in box_list]
            cur_count = save_patch_image(save_dir, image_list, valid_list, cur_count)


def init_box(mode, num, image_width, image_height, patch_size, overlay=0.3):
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
        x_iteration = int(image_width // stride) - 1
        y_iteration = int(image_height // stride) - 1

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

divide_dataset('images','temp_data/pitches/train','temp_data/pitches/test', 24)
print("finish!")