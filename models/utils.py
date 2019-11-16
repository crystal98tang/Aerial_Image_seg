# If can not work, add "tensorflow.python."
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *

"""
    ShortCutBlock Layer
"""
def shortcutblock(filter):
    def _create_shortcut_block(inputs):
        conv_main = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)
        conv_main = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(conv_main)
        conv_main = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv_main)
        conv_main = BatchNormalization()(conv_main)

        conv_fine = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)
        conv_fine = BatchNormalization()(conv_fine)

        merge = Add()([conv_main, conv_fine])

        conv = Activation('relu')(merge)

        return conv
    return _create_shortcut_block