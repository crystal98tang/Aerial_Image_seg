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
    model = FCN_8()
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
        # FIXME:file读取尚未完成
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
        model.fit_generator(myGene, steps_per_epoch=steps, epochs=itrs, verbose=2,
                            callbacks=[model_checkpoint, TensorBoard(log_dir=log_dir,
                                                                     write_grads=1, write_images=1)],
                            shuffle=True, workers=2, use_multiprocessing=True)
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
    parallel_model.fit_generator(myGene, steps_per_epoch=steps, epochs=itrs, verbose=2,
                                 callbacks=[model_checkpoint, TensorBoard(log_dir=log_dir,
                                                                          write_grads=1, write_images=1)],
                                 shuffle=True, workers=2, use_multiprocessing=True)  #

elif run_mode == "test":
    model.load_weights(saved_model)
    image_path = os.path.join(valid_data_dir, "image")
    label_path = os.path.join(valid_data_dir, "label")
    imagelist = os.listdir(image_path)
    labellist = os.listdir(label_path)
    if test_mode == 'auto':
        sum_image = imagelist.__len__()
        batch_image = 50
        itr = sum_image // batch_image
    elif test_mode == 'manual':
        # 手动部分
        batch_image = global_test_image_num
        itr = global_test_itr
    else:
        print("test mode choose error")
        os._exit(1)

    if not os.path.exists(saved_results_path):
        os.makedirs(saved_results_path)
    #
    eval_p = {'Recall': [], 'Prescision': [], 'F_measure': [], 'IoU': [], 'Dice': []}
    eval_oc = {'Recall': [], 'Prescision': [], 'F_measure': [], 'IoU': [], 'Dice': []}
    eval_co = {'Recall': [], 'Prescision': [], 'F_measure': [], 'IoU': [], 'Dice': []}
    #
    sum_pixel = global_image_size * global_image_size
    for i in range(itr):
        testGene = testGenerator(imagelist, i * batch_image, image_path, batch_image)
        results = model.predict_generator(testGene, batch_image, verbose=1)
        # use_multiprocessing=True` is not supported on Windows
        tmp = 0

        for k in range(i * batch_image, (i + 1) * batch_image):
            #
            image = np.array(imageio.imread(os.path.join(image_path, imagelist[k])))
            #
            label = np.array(imageio.imread(os.path.join(label_path, labellist[k])))

            label_01 = label / 255  # 归一化

            # 全黑无效 && <5%无效
            if label_01.max() != 1:
                tmp += 1
                continue
            tmp_pixel = np.count_nonzero(label_01)
            if (tmp_pixel / sum_pixel) < 0.05:
                tmp += 1
                continue
            # pred score
            res = results[tmp][:, :, 1]
            tmp += 1
            # 阈值
            th = 1 - 0.618
            # 单CRF
            # crf_res = crf.CRFs(image, res)
            #############################
            # 形态学开闭
            res_mor_oc = morph.morph(res, operation='oc', vary=True, th=th)
            #############################
            # 形态学闭开
            res_mor_co = morph.morph(res, operation='co', vary=True, th=th)
            #############################
            res = vary(res, th)
            # 全黑无效
            if res.max() != 1:
                continue
            # # Recall
            recall_orgin = eva.Recall(res, label_01)
            eval_p['Recall'].append(recall_orgin)
            recall_oc = eva.Recall(res_mor_oc, label_01)  # 结果不好
            eval_oc['Recall'].append(recall_oc)
            recall_co = eva.Recall(res_mor_co, label_01)
            eval_co['Recall'].append(recall_co)
            # # Percision
            prec_orgin = eva.Precision(res, label_01)
            eval_p['Prescision'].append(prec_orgin)
            prec_oc = eva.Precision(res_mor_oc, label_01)  # 结果不好
            eval_oc['Prescision'].append(prec_oc)
            prec_co = eva.Precision(res_mor_co, label_01)
            eval_co['Prescision'].append(prec_co)
            # # F-score
            F_orgin = eva.F_measure(recall_orgin, prec_orgin)
            eval_p['F_measure'].append(F_orgin)
            F_oc = eva.F_measure(recall_oc, prec_oc)  # 结果不好
            eval_oc['F_measure'].append(F_oc)
            F_co = eva.F_measure(recall_co, prec_co)
            eval_co['F_measure'].append(F_co)
            # IoU
            IoU_orgin = eva.mean_iou(res, label_01)
            eval_p['IoU'].append(IoU_orgin)
            IoU_oc = eva.mean_iou(res_mor_oc, label_01)  # 结果不好
            eval_oc['IoU'].append(IoU_oc)
            IoU_co = eva.mean_iou(res_mor_co, label_01)
            eval_co['IoU'].append(IoU_co)
            # dice
            dice_orgin = eva.dice(res, label_01)
            eval_p['Dice'].append(dice_orgin)
            dice_oc = eva.dice(res_mor_oc, label_01)  # 结果不好
            eval_oc['Dice'].append(dice_oc)
            dice_co = eva.dice(res_mor_co, label_01)
            eval_co['Dice'].append(dice_co)

            print(imagelist[k])
            print("-" * 10)

            # Out
            imageio.imwrite(os.path.join(saved_results_path, "%d_image.tif" % k), image)
            imageio.imwrite(os.path.join(saved_results_path, "%d_gt.tif" % k), label)
            imageio.imwrite(os.path.join(saved_results_path, "%d_2_predict.tif" % k), res)
            imageio.imwrite(os.path.join(saved_results_path, "%d_2_mor_oc_predict.tif" % k), res_mor_oc) # 结果不好
            imageio.imwrite(os.path.join(saved_results_path, "%d_2_mor_co_predict.tif" % k), res_mor_co)
            # imageio.imwrite(os.path.join(saved_results_path, "%d_2_crf_predict.tif" % k), crf)

    #均值方差
    eva.mean_variance(eval_p)
    eva.mean_variance(eval_p)
    eva.mean_variance(eval_p)
    #
    import GUI.result as ts
    ts.box_show(eval_p)
    ts.box_show(eval_oc)
    ts.box_show(eval_co)
    # print(sum_1 / (batch_image - out))
    # print(sum_2 / (batch_image - out))
    # print(sum_3 / (batch_image - out))
    #均值方差

    #
    # if save_mode == "single":
    #     saveResult("temp_data/temp_test", results)
    # elif save_mode == "full":
    #     saveBigResult("temp_data/full", results, init_box, each_image_size, num)


def toSaveImage(saved_results_path, image, name, th):
    image[image > th] = 1
    image[image < th] = 0
    imageio.imwrite(os.path.join(saved_results_path, "%d_2_%s.tif" % (i, name)), image[:, :, 0])
