from utils import *

# choose model
model_name = "SDFCN"
# timestamp
time_now = time.strftime("_%Y_%m_%d__%H_%M")
print("*" * 10 + time_now + "*" * 10)
# itrs & steps
itrs = 10
steps = 200
batchs = 3
print(str(itrs) + ' ' + str(steps) + ' ' + str(batchs))
# Choose Devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
run_mode = "train"
mult_thread = False
#
save_mode = "single"
# Image Setting
dataset = 'IAILD'   # select dataset
read_image_mode = 'path'  # from path or file to read images
global_image_size = 512     # source image size
global_image_channels = 3   # source image channels (RGB = 3/RGBN = 4)
global_image_mode = 'rgb'   # source image mode (rgb / ?)
global_label_classes = 2    # label classes
global_label_mode = 'grayscale'  # label image mode (grayscale / rgb)
# Image Path
if dataset == 'IAILD':
    train_data_dir = "DataSet/IAILD/train"
    valid_data_dir = "DataSet/IAILD/valid"
elif dataset == 'GID':
    src_image_data_dir = "DataSet/IAILD/image_RGB"
    src_label_data_dir = "DataSet/IAILD/label_5classes"
    train_data_dir = "DataSet/IAILD/train_RGB"
    valid_data_dir = "DataSet/IAILD/vaild_RGB"
    label_data_dir = "DataSet/IAILD/label_RGB"
#
log_dir = "./logs/log_" + model_name + time_now
#
saved_model = model_name + ".hdf5"
#
print(saved_model)
#

# training images preprocess args
data_gen_args = dict(rotation_range=45,  # 随机角度 应为整数
                     # width_shift_range=0.05, #水平平移
                     # height_shift_range=0.05,#垂直平移
                     # shear_range=0.05,
                     # zoom_range=0.05,
                     horizontal_flip=True,  # 随机水平翻转
                     vertical_flip=True,  # 随机垂直翻转
                     brightness_range=[0, 1],  # 选择亮度偏移值的范围。
                     fill_mode='nearest')
