import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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


def box_show(dict, path, title):
    #
    df = pd.DataFrame(dict)  # 装入DataFrame中
    df.boxplot()  # 也可用plot.box()
    plt.title(title, fontsize=12, color='black')
    plt.savefig(os.path.join(path, "%s.jpg" % title))
#     plt.show()
    plt.close('all')