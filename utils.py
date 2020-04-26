# If can not work, add "tensorflow.python."
# from tensorflow.python.keras.callbacks import ModelCheckpoint,TensorBoard
# from tensorflow.python.keras.utils import multi_gpu_model
# from tensorflow.python.keras.utils import multi_gpu_model
#
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.utils import multi_gpu_model
from keras.utils import multi_gpu_model

import keras.backend as K
'''
'''
import numpy as np
import time
import os
import cv2
import imageio
import tensorflow as tf
import postprocess.crf as crf
import evaluate.utils as eva
import postprocess.Morphological as morph
from postprocess.merge import merge_list_big, merge_single_big


from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, classification_report

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


#精确率评价指标
def metric_precision(y_true,y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32'))

    precision = true_positives / (pred_positives + K.epsilon())
    return precision


#召回率评价指标
def metric_recall(y_true,y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.20), 'float32'))

    recall = true_positives / (poss_positives + K.epsilon())
    return recall


#F1-score评价指标
def metric_F1score(y_true,y_pred):
    p_val = metric_precision(y_true, y_pred)
    r_val = metric_recall(y_true, y_pred)
    f_val = 2 * p_val * r_val / (p_val + r_val)

    return f_val

# dice
def dice_ratio(labels,preds):
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

def vary_plus(img):
    return 0

def toSaveImage(saved_results_path, image, name, i,th):
    image = vary(image, th)
    imageio.imwrite(os.path.join(saved_results_path, "%d_2_%s.tif" % (i, name)), image)


def file_exist(run_mode, path):
    if os.path.exists(path) and run_mode!='test':
        if input("Already exist model file, Overwrite?") == "Y":
            pass
        else:
            exit(1)