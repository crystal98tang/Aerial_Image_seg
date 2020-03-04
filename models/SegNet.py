from models.utils import *


def SegNet(input_size=(imageSize, imageSize, Channels)):

    inputs = Input(input_size)
    # encoder
    conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    conv = BatchNormalization()(conv)
    conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)

    conv = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    # (64,64)
    conv = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    # (32,32
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    # (16,16
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    # (8,8
    # decode
    conv = UpSampling2D(size=(2, 2))(conv)
    # (16,16
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = UpSampling2D(size=(2, 2))(conv)
    # (32,32
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = UpSampling2D(size=(2, 2))(conv)
    # (64,64
    conv = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = UpSampling2D(size=(2, 2))(conv)

    conv = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = UpSampling2D(size=(2, 2))(conv)

    conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(Classes, (1, 1), strides=(1, 1), padding='same')(conv)

    conv = Activation('softmax')(conv)
    model = Model(inputs, conv)

    return model
