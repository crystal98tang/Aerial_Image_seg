import numpy as np


def RecPreF(predict, label, size):
    # 0 / 1 二分图
    B_seg = 0
    I_unseg = 0
    I_wseg = 0
    #
    height = size
    weight = size
    #
    for x in range(0, height):
        for y in range(0, weight):
            if predict[x, y] == 1 and label[x, y] == 1:
                B_seg += 1
            elif label[x, y] == 1 and predict[x, y] == 0:
                I_unseg += 1
            elif label[x, y] == 0 and predict[x, y] == 1:
                I_wseg += 1
    try:
        recall = B_seg / (B_seg + I_unseg)
        precision = B_seg / (B_seg + I_wseg)
        F = 2.0 * recall * precision / (recall + precision)
    except ZeroDivisionError:
        print("Warning: Recall_Precision_F error")
        recall = 0
        precision = 0
        F = 0.0
    print("recall:%.2f%%" % (recall * 100))
    print("precision:%.2f%%" % (precision * 100))
    print("F:%.2f%%" % (F * 100))
    print("---")
    return recall, precision, F


def IouDice(y_pred, y_true):
    interArea = np.multiply(y_true, y_pred)  # 交集
    tem = y_true + y_pred
    unionArea = tem - interArea  # 并集

    w1 = np.sum(interArea)
    w2 = np.sum(unionArea)

    size_i1 = np.count_nonzero(y_pred)
    size_i2 = np.count_nonzero(y_true)

    try:
        IoU = w1 / w2
    except ZeroDivisionError:
        print("Warning: IoU error")
        IoU = 0

    try:
        dice = w1 * 2. / float(size_i1 + size_i2)
    except ZeroDivisionError:
        print("Warning: dice error")
        dice = 0

    print("IoU:%.2f%%" % (IoU * 100))
    print("Dice:%.2f%%" % (dice * 100))
    print("---")
    return IoU, dice


def Recall(predict, label, size):
    # 0 / 1 二分图
    B_seg = 0
    I_unseg = 0
    #
    height = size
    weight = size
    #
    for x in range(0, height):
        for y in range(0, weight):
            if predict[x, y] == 1 and label[x, y] == 1:
                B_seg += 1
            elif label[x, y] == 1 and predict[x, y] == 0:
                I_unseg += 1
    try:
        recall = B_seg / (B_seg + I_unseg)
    except ZeroDivisionError:
        print("Warning: Recall error")
        recall = 0
    print("recall:%.2f%%" % (recall * 100))
    return recall


def Precision(predict, label, size):
    # 0 / 1 二分图
    B_seg = 0
    I_wseg = 0
    #
    height = size
    weight = size
    #
    for x in range(0, height):
        for y in range(0, weight):
            if predict[x, y] == 1 and label[x, y] == 1:
                B_seg += 1
            elif label[x, y] == 0 and predict[x, y] == 1:
                I_wseg += 1
    try:
        precision = B_seg / (B_seg + I_wseg)
    except ZeroDivisionError:
        print("Warning: precision error")
        precision = 0
    print("precision:%.2f%%" % (precision * 100))
    return precision


def F_measure(recall, precision):
    try:
        F = 2.0 * recall * precision / (recall + precision)
    except ZeroDivisionError:
        print("Warning: F error")
        F = 0.0
    print("F:%.2f%%" % (F * 100))
    return F


def mean_iou(y_pred, y_true):
    interArea = np.multiply(y_true, y_pred)  # 交集
    tem = y_true + y_pred
    unionArea = tem - interArea  # 并集
    w1 = np.sum(interArea)
    w2 = np.sum(unionArea)
    try:
        IoU = w1 / w2
    except ZeroDivisionError:
        print("Warning: IoU error")
        IoU = 0
    print("IoU:%.2f%%" % (IoU * 100))
    return IoU


def dice(y_pred, y_true):
    interArea = np.multiply(y_true, y_pred)  # 交集
    w1 = np.sum(interArea)
    size_i1 = np.count_nonzero(y_pred)
    size_i2 = np.count_nonzero(y_true)
    try:
        dice = w1 * 2. / float(size_i1 + size_i2)
    except ZeroDivisionError:
        print("Warning: dice error")
        dice = 0
    print("Dice_new:%.2f%%" % (dice * 100))
    return dice


def dc(input1, input2):
    """
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC = \frac{2|A\capB|}{|A|+|B|}

    , where A is the first and B the second set of samples (here binary objects).

    Parameters
    ----------
    input1: array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    input2: array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc: float
        The Dice coefficient between the object(s) in `input1` and the
        object(s) in `input2`. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric.
    """
    input1 = np.atleast_1d(input1.astype(np.bool))
    input2 = np.atleast_1d(input2.astype(np.bool))

    intersection = np.count_nonzero(input1 & input2)

    size_i1 = np.count_nonzero(input1)
    size_i2 = np.count_nonzero(input2)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    print("Dice_old:%.5f%%" % dc)
    return dc


def mean_variance(list):
    for i in list:
        v = np.var(list[i])
        avg = np.average(list[i])
        print("%s:%.3f±%.3f" % (i, avg, v))
    print('-' * 10)
