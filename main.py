from args import *
from data.dataset import *

from models.SDFCN import *
from models.Unet import *

if model_name == "SDFCN":
    model = SDFCN()
elif model_name == "Unet":
    model = Unet()

if run_mode == "train":
    myGene = trainGenerator(5, train_data_dir,data_gen_args,save_to_dir = None)
    model_checkpoint = ModelCheckpoint(filepath=saved_model, monitor='loss',verbose=1, save_best_only=False)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(myGene,steps_per_epoch=100, epochs=5,verbose=2,
                        callbacks=[model_checkpoint,TensorBoard(log_dir=log_dir,
                                                                write_grads=1,write_images=1)],
                        shuffle=True) #,workers=2, use_multiprocessing=False
elif run_mode == "train_GPUs":
    myGene = trainGenerator(5, train_data_dir, data_gen_args, save_to_dir=None)
    parallel_model = multi_gpu_model(model, gpus=2)
    checkpoint = ParallelModelCheckpoint(model, filepath=saved_model)  # 解决多GPU运行下保存模型报错的问题
    model_checkpoint = ModelCheckpoint(filepath=saved_model, monitor='loss', verbose=1, save_best_only=False)
    parallel_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    parallel_model.fit_generator(myGene,steps_per_epoch=10,epochs=2,verbose=2,
                                 callbacks=[model_checkpoint,TensorBoard(log_dir=log_dir,
                                                        write_grads=1,write_images=1)],
                                 shuffle=True, workers=2, use_multiprocessing=True) #
elif run_mode == "test":
    model.load_weights(saved_model)

#
