from models.utils import *

from tensorflow.python.keras.regularizers import l2
from models.BilinearUpSampling import *


def FCN_8(nClasses=Classes, input_height=imageSize, input_width=imageSize, nChannels=Channels):
    inputs = Input((input_height, input_width, nChannels))

    conv1 = Conv2D(filters=32, input_shape=(input_height, input_width, nChannels),
                   kernel_size=(3, 3), padding='same', activation='relu',
                   name='block1_conv1')(inputs)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block1_conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block2_conv1')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block2_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv1')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv2')(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)
    # 8倍下采样结果
    score_pool3 = Conv2D(filters=2, kernel_size=(3, 3), padding='same',
                         activation='relu', name='score_pool3')(pool3)

    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv1')(pool3)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv2')(conv4)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv3')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(conv4)
    # 16倍下采样结果
    score_pool4 = Conv2D(filters=2, kernel_size=(3, 3), padding='same',
                         activation='relu', name='score_pool4')(pool4)

    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv1')(pool4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv2')(conv5)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv3')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='block5_pool')(conv5)

    fc6 = Conv2D(filters=1024, kernel_size=(1, 1), padding='same', activation='relu',
                 name='fc6')(pool5)
    fc6 = Dropout(0.3, name='dropout_1')(fc6)

    fc7 = Conv2D(filters=1024, kernel_size=(1, 1), padding='same', activation='relu',
                 name='fc7')(fc6)
    fc7 = Dropout(0.3, name='dropour_2')(fc7)

    # 32倍下采样结果
    score_fr = Conv2D(filters=nClasses, kernel_size=(1, 1), padding='same',
                      activation='relu', name='score_fr')(fc7)
    # Conv2DTranspose转置卷积
    score2 = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None,
                             name="score2")(score_fr)

    # 32倍下采样结果和16倍下采样结果相加
    add1 = add(inputs=[score2, score_pool4], name="add_1")

    score4 = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None,
                             name="score4")(add1)
    # 32倍下采样结果和16倍下采样结果相加后与8倍下采样率
    add2 = add(inputs=[score4, score_pool3], name="add_2")
    # 使用转置卷积还原到原图大小的语义分割结果
    UpSample = Conv2DTranspose(filters=nClasses, kernel_size=(8, 8), strides=(8, 8),
                               padding="valid", activation=None,
                               name="UpSample")(add2)

    outputs = core.Activation('softmax')(UpSample)

    model = Model(inputs, outputs)

    return model


