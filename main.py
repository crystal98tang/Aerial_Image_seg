from args import *
from data.dataset import *


model = None
if model_name == "SDFCN":
    from models.SDFCN import *
    model = SDFCN()
elif model_name == "Unet":
    from models.Unet import *
    model = Unet()
elif model_name == "FCN":
    from models.FCN import *
    model = FCN_Vgg16_32s()
elif model_name == "DenseFCN":
    from models.DenseFCN import *
    # model = DenseNet_FCN()
elif model_name == "SegNet":
    from models.SegNet import *
    model = SegNet()

if model is None:
    print("Model choose error")
    os._exit(1)

if run_mode == "train":
    # Generator
    myGene = None
    if read_image_mode == 'path':
        myGene = trainGenerator_path(
            batchs, train_data_dir, data_gen_args, image_color_mode=args.global_image_mode,
            mask_color_mode=args.global_label_mode, target_size=(args.global_image_size, args.global_image_size),
            classes=global_label_classes)
    elif read_image_mode == 'file':
        #FIXME:file读取尚未完成
        myGene = trainGenerator_file(
            batchs, train_data_dir, data_gen_args, image_color_mode=args.global_image_mode,
            mask_color_mode=args.global_label_mode, target_size=(args.global_image_size, args.global_image_size))
    # Tensorboard
    model_checkpoint = ModelCheckpoint(
        filepath=saved_model, monitor='loss', verbose=1, save_best_only=False
    )
    # model compile
    model.compile(
        optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy']
    )
    if mult_thread:
        model.fit_generator(myGene,steps_per_epoch=steps, epochs=itrs,verbose=2,
                            callbacks=[model_checkpoint,TensorBoard(log_dir=log_dir,
                                                                    write_grads=1,write_images=1)],
                            shuffle=True,workers=2, use_multiprocessing=True)
    else:
        model.fit_generator(myGene, steps_per_epoch=steps, epochs=itrs, verbose=2,
                            callbacks=[model_checkpoint, TensorBoard(log_dir=log_dir,
                                                                     write_grads=1, write_images=1)],
                            shuffle=True, use_multiprocessing=False)

elif run_mode == "train_GPUs":
    myGene = trainGenerator(batchs, train_data_dir, data_gen_args, save_to_dir=None)
    parallel_model = multi_gpu_model(model, gpus=2)
    checkpoint = ParallelModelCheckpoint(model, filepath=saved_model)  # 解决多GPU运行下保存模型报错的问题
    model_checkpoint = ModelCheckpoint(filepath=saved_model, monitor='loss', verbose=1, save_best_only=False)
    parallel_model.compile(optimizer=Adam(lr=1.0e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    parallel_model.fit_generator(myGene,steps_per_epoch=steps,epochs=itrs,verbose=2,
                                 callbacks=[model_checkpoint,TensorBoard(log_dir=log_dir,
                                                        write_grads=1,write_images=1)],
                                 shuffle=True, workers=2, use_multiprocessing=True) #

elif run_mode == "test":
    model.load_weights(saved_model)
    image_path = os.path.join(valid_data_dir, "image")
    label_path = os.path.join(valid_data_dir, "label")
    imagelist = os.listdir(image_path)
    labellist = os.listdir(label_path)
    # num_image = imagelist.__len__()
    num_image = 100
    testGene = testGenerator(imagelist, image_path, num_image)
    results = model.predict_generator(testGene, num_image, verbose=1)
    # use_multiprocessing=True` is not supported on Windows
    # TODO:后处理
    import postprocess.test as post
    # import postprocess.crf as crf
    os.makedirs(saved_results_path)
    out = 0
    sum_1 = sum_2 = sum_3 = 0.0
    for i in range(num_image):
        #
        image = np.array(imageio.imread(os.path.join(image_path, imagelist[i])))
        #
        label = np.array(imageio.imread(os.path.join(label_path, labellist[i])))
        imageio.imwrite(os.path.join(saved_results_path, "%d_gt.tif" % i), label)
        label = label / 255 # 归一化
        # pred score
        res = results[i]
        # 单CRF
        # crf_res = crf.CRFs(image, res)
        # imageio.imwrite(os.path.join(saved_results_path, "%d_crf.tif" % i), image)
        #############################
        # 变整数
        th = 0.5
        # 反色
        res = 1 - res
        ############################
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k_itr = 3
        # 单形态学变换 先开后闭
        xt_oc_res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel, iterations=k_itr)
        xt_oc_res = cv2.morphologyEx(xt_oc_res, cv2.MORPH_CLOSE, kernel, iterations=k_itr)
        xt_oc_res[xt_oc_res > th] = 1
        xt_oc_res[xt_oc_res < th] = 0
        xt_oc_res = xt_oc_res[:, :, 0]
        imageio.imwrite(os.path.join(saved_results_path, "%d_2_xt_oc_res.tif" % i), xt_oc_res)
        # 单形态学变换 先闭后开
        xt_co_res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=3)
        xt_co_res = cv2.morphologyEx(xt_co_res, cv2.MORPH_OPEN, kernel, iterations=3)
        xt_co_res[xt_co_res > th] = 1
        xt_co_res[xt_co_res < th] = 0
        xt_co_res = xt_co_res[:, :, 0]
        imageio.imwrite(os.path.join(saved_results_path, "%d_2_xt_co_res.tif" % i), xt_co_res)
        ############################
        # # CRF+形态学 先开后闭
        # crf_xt_oc_res = cv2.morphologyEx(crf_res, cv2.MORPH_OPEN, kernel, iterations=3)
        # crf_xt_oc_res = cv2.morphologyEx(crf_xt_oc_res, cv2.MORPH_CLOSE, kernel, iterations=3)
        # # CRF+形态学 先闭后开
        # crf_xt_co_res = cv2.morphologyEx(crf_res, cv2.MORPH_CLOSE, kernel, iterations=3)
        # crf_xt_co_res = cv2.morphologyEx(crf_xt_co_res, cv2.MORPH_OPEN, kernel, iterations=3)
        # ###########################
        # # 形态学 先开后闭+CRF
        # xt_crf_oc_res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel, iterations=3)
        # xt_crf_oc_res = cv2.morphologyEx(xt_crf_oc_res, cv2.MORPH_CLOSE, kernel, iterations=3)
        # crf_res = crf.CRFs(image, res)

        # dice
        res[res > th] = 1
        res[res < th] = 0
        res = res[:, :, 0]
        dice_res = post.dc(label, res)
        dice_1 = post.dc(label, xt_oc_res)
        dice_2 = post.dc(label, xt_co_res)
        if dice_res >= 0.3 and dice_1 >= 0.3 and dice_2 >= 0.3:
            sum_1 += dice_res
            sum_2 += dice_1
            sum_3 += dice_2
        else:
            print(i)
            out += 1
        # IoU
        # print(post.mean_iou(label, res))
        #
        imageio.imwrite(os.path.join(saved_results_path, "%d_image.tif" % i), image)
        imageio.imwrite(os.path.join(saved_results_path, "%d_2_predict.tif" % i), res)

    print(sum_1 / (num_image - out))
    print(sum_2 / (num_image - out))
    print(sum_3 / (num_image - out))

    #
    # if save_mode == "single":
    #     saveResult("temp_data/temp_test", results)
    # elif save_mode == "full":
    #     saveBigResult("temp_data/full", results, init_box, each_image_size, num)

def toSaveImage(saved_results_path, image, name, th):
    image[image > th] = 1
    image[image < th] = 0
    imageio.imwrite(os.path.join(saved_results_path, "%d_2_%s.tif" % (i, name)), image[:, :, 0])