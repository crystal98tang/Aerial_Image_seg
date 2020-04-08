import numpy as np


def merge_list_big(mark_big, imgs, y, overlapRate, size):
    """
    融合
    :param imgs: 一行概率图瓦片
    :param overlapRate:
    :param size:
    :return:
    """
    # stride and iteration
    stride = size * (1 - overlapRate)
    i = 0
    for img in imgs:
        for y_small in range(0, 256):
            for x_small in range(0, 256):
                cur_x = int(i * stride + x_small)
                cur_y = int(y * stride + y_small)
                cur_pixel = mark_big[cur_y][cur_x][:]
                new_pixel_val = img[y_small][x_small][:]
                if cur_pixel.all() == 0:
                    mark_big[cur_y][cur_x] = new_pixel_val
                else:
                    mark_big[cur_y][cur_x] = [np.average([cur_pixel[0], new_pixel_val[0]]),
                                              np.average([cur_pixel[1], new_pixel_val[1]])] # 求平均
        i += 1
    return mark_big


def merge_single_big(mark_big, imgs, y, overlapRate, size):
    """
    融合
    :param imgs: 一行概率图瓦片
    :param overlapRate:
    :param size:
    :return:
    """
    # stride and iteration
    stride = size * (1 - overlapRate)
    i = 0
    for img in imgs:
        for y_small in range(0,256):
            for x_small in range(0,256):
                cur_x = (int) (i * stride + x_small)
                cur_y = (int) (y * stride + y_small)
                cur_pixel = mark_big[cur_x][cur_y][0]
                new_pixel_val = img[x_small][y_small][1]
                if cur_pixel == 0:
                    mark_big[cur_x][cur_y][0] = new_pixel_val
                else:
                    mark_big[cur_x][cur_y][0] = np.average([cur_pixel,new_pixel_val]) # 求平均
        i += 1
    return mark_big