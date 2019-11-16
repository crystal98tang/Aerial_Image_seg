from utils import *
time_now = time.strftime("%Y_%m_%d__%H_%M")
print("*"*10 + time_now + "*"*10)
#

## Choose Devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
run_mode = "train"
#
save_mode = "single"
#
train_data_dir = "DataSet/train"
#
valid_data_dir = "DataSet/valid"
#
log_dir = "./logs/log_" + time_now
#
saved_model = "SDFCN.hdf5"
#
print(saved_model)
# training images preprocess args
data_gen_args = dict(rotation_range=45,    #随机角度 应为整数
                    #width_shift_range=0.05, #水平平移
                    #height_shift_range=0.05,#垂直平移
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,   #随机水平翻转
                    vertical_flip=True,     #随机垂直翻转
                    brightness_range=[0,1],  #选择亮度偏移值的范围。
                    fill_mode='nearest')