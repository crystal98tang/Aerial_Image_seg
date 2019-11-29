from utils import *

# choose model
model_name = "FCN"

# timestamp
time_now = time.strftime("_%Y_%m_%d__%H_%M")
print("*"*10 + time_now + "*"*10)
#itrs & steps
itrs = 5e3
steps = 150
batchs = 10
## Choose Devices
GPUs_num = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
run_mode = "train_GPUs"
#
save_mode = "single"
#
train_data_dir = "DataSet/train"
#
valid_data_dir = "DataSet/valid"
#
log_dir = "./logs/log_" + model_name + time_now
#
saved_model = model_name + ".hdf5"
#
print(saved_model)
# training images preprocess args
data_gen_args = dict(rotation_range=45,    #随机角度 应为整数
                    #width_shift_range=0.05, #水平平移
                    #height_shift_range=0.05,#垂直平移
                    # shear_range=0.05,
                    # zoom_range=0.05,
                    horizontal_flip=True,   #随机水平翻转
                    vertical_flip=True,     #随机垂直翻转
                    brightness_range=[0,1],  #选择亮度偏移值的范围。
                    fill_mode='nearest')