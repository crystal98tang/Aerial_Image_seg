# If can not work, add "tensorflow.python."
from tensorflow.python.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.utils import multi_gpu_model
#
# from keras.callbacks import ModelCheckpoint,TensorBoard
# from keras.utils import multi_gpu_model
# from keras.utils import multi_gpu_model
'''
'''
import numpy as np
import time
import os
import cv2
import imageio

import postprocess.crf as crf
import evaluate.utils as eva
import postprocess.Morphological as morph
#
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)
# loss function
def mult_loss(preds, labels):
    return

# dice
def dice_ratio(preds, labels):
    '''
    preds & labels should only contain 0 or 1.
    '''
    return np.sum(preds[labels==1])*2.0 / (np.sum(preds) + np.sum(labels))

# # 加载器
# num = 20
# days = 365
# for i in range(days):
#     print("\r","加载中：{0}%".format(round((i+1)*100 / days)),end='',flush=True)
#     time.sleep(0.06)
# # 库tqdm
# # 库progressbar


def vary(img, th):
    img[img > th] = 1
    img[img < th] = 0
    return img


def toSaveImage(saved_results_path, image, name, i,th):
    image = vary(image, th)
    imageio.imwrite(os.path.join(saved_results_path, "%d_2_%s.tif" % (i, name)), image)


def file_exist(run_mode, path):
    if os.path.exists(path) and run_mode!='test':
        if input("Already exist model file, Overwrite?") == "Y":
            pass
        else:
            exit(1)