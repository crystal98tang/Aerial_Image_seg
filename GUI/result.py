import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def easy_show(
        num = None,
        Recall = None,
        Prescision = None,
        F_measure = None,
        IoU = None,
        Dice = None):
    #
    num = range(len(Dice))
    plt.plot(num, Recall, mec='r', mfc='w', label='Recall')
    plt.plot(num, Prescision, mec='b', mfc='w', label='Prescision')
    plt.plot(num, F_measure, mec='y', mfc='w', label='F_measure')
    plt.plot(num, IoU, mec='g', mfc='w', label='IoU')
    plt.plot(num, Dice, mec='black', mfc='w', label='Dice')
    plt.legend()
    plt.show()


def box_show(
        num = None,
        Recall = None,
        Prescision = None,
        F_measure = None,
        IoU = None,
        Dice = None):
    #
    df = pd.DataFrame({'Recall':Recall,'Prescision':Prescision,'F_measure':F_measure,'IoU':IoU,'Dice':Dice})  # 先生成0-1之间的5*4维度数据，再装入4列DataFrame中
    df.boxplot()  # 也可用plot.box()
    plt.show()
