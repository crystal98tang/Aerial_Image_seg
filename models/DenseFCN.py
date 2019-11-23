# from models.utils import *
# from keras_contrib.applications import densenet
# from keras_applications.imagenet_utils import _obtain_input_shape
# from tensorflow.python.keras.regularizers import l2
#
# def top(x, input_shape, classes, activation, weight_decay):
#
#     x = Conv2D(classes, (1, 1), activation='linear',
#                padding='same', kernel_regularizer=l2(weight_decay),
#                use_bias=False)(x)
#
#     if K.image_data_format() == 'channels_first':
#         channel, row, col = input_shape
#     else:
#         row, col, channel = input_shape
#
#     if activation is 'sigmoid':
#         x = Reshape((row * col * classes,))(x)
#
#     return x
#
# def DenseNet_FCN(input_shape=None, weight_decay=1e-4, classes=2, activation='softmax',include_top=False):
#     input_shape = _obtain_input_shape(input_shape,
#                                       default_size=32,
#                                       min_size=16,
#                                       data_format=K.image_data_format()
#                                       ,require_flatten=False
#                                       )
#     img_input = Input(shape=input_shape)
#
#     x = densenet.__create_fcn_dense_net(classes, img_input,
#                                         input_shape=input_shape,
#                                         nb_layers_per_block=[4, 5, 7, 10, 12, 15],
#                                         growth_rate=16,
#                                         dropout_rate=0.2,
#                                         include_top=include_top)
#
#     x = top(x, input_shape, classes, activation, weight_decay)
#
#     model = Model(img_input, x, name='DenseNet_FCN')
#     return model
