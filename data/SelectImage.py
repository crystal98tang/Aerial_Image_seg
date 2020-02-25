import imageio
import numpy as np
import os

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
KindOfImage = [[], [], [], [], []]


def select_image(src_label_path, src_image_path):
    files_label = os.listdir(src_label_path)
    print("find:" + str(files_label.__len__()))
    for f in files_label:
        if f.endswith('.tif'):
            print(os.path.join(src_label_path, f))
            cnt = [0, 0, 0, 0, 0]
            img_path = os.path.join(src_label_path, f)
            img = imageio.imread(img_path)
            img //= 255
            height = img.shape[0]
            weight = img.shape[1]
            # sampling step is 10.
            for x in range(0, height, 10):
                for y in range(0, weight, 10):
                    if img[x, y, 0] == 1:
                        if img[x, y, 1] == 1:
                            cnt[4] += 1
                        else:
                            cnt[3] += 1
                    else:
                        if img[x, y, 1] == 1:
                            if img[x, y, 2] == 1:
                                cnt[2] += 1
                            else:
                                cnt[1] += 1
                        else:
                            if img[x, y, 2] == 1:
                                cnt[0] += 1
            print(cnt)
            choose = cnt.index(max(cnt))
            print(choose)
            KindOfImage[choose].append(img_path)

            # To rename
            new_file_name = str(choose) + '_' + str(KindOfImage[choose].__len__()) + '.tif'
            # rename label
            label_path = os.path.join(src_label_path, f)
            new_file_path = os.path.join(src_label_path, new_file_name)
            os.rename(label_path, new_file_path)
            # rename image
            file_name = f[:-10] + '.tif'    # label和image名不一样 少_label
            image_path = os.path.join(src_image_path, file_name)
            new_file_path = os.path.join(src_image_path, new_file_name)
            os.rename(image_path, new_file_path)


def save_info(list_file_path):
    i = 0
    for name in KindOfName:
        with open(os.path.join(list_file_path, "%s.txt" % name), "w") as f:
            f.writelines(KindOfImage[i])
            i += 1


def read_info(list_file_path):
    i = 0
    for name in KindOfName:
        with open(os.path.join(list_file_path, "%s.txt" % name), "r") as f:
            KindOfImage[i].append(f.readlines())
            i += 1

select_image('../DataSet/GID/label_5classes','../DataSet/GID/image_RGB')
save_info('../DataSet/GID/train_RGB')
for i in range(KindOfImage.__len__()):
    print(KindOfImage[i].__len__(), end=' ')

# read_info('../DataSet/GID/train_RGB')
print("over")
