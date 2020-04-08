from args import *
from data.dataset import *

model = None
if model_name == "SDFCN":
    from models.SDFCN import *
    model = SDFCN()
elif model_name == "MRDFCN":
    from models.MRDFCN import *
    model = MRDFCN()
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
# TODO: no GPUs drivers
# elif run_mode == "train_GPUs":
    # myGene = trainGenerator(batchs, train_data_dir, data_gen_args, save_to_dir=None)
    # parallel_model = multi_gpu_model(model, gpus=2)
    # checkpoint = ParallelModelCheckpoint(model, filepath=saved_model)  # 解决多GPU运行下保存模型报错的问题
    # model_checkpoint = ModelCheckpoint(filepath=saved_model, monitor='loss', verbose=1, save_best_only=False)
    # parallel_model.compile(optimizer=Adam(lr=1.0e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    # parallel_model.fit_generator(myGene, steps_per_epoch=steps, epochs=itrs, verbose=2,
    #                              callbacks=[model_checkpoint, TensorBoard(log_dir=log_dir,
    #                                                                       write_grads=1, write_images=1)],
    #                              shuffle=True, workers=2, use_multiprocessing=True)  #

elif run_mode == "test":
    model.load_weights(saved_model)
    image_path = os.path.join(valid_data_dir, "image")
    label_path = os.path.join(valid_data_dir, "label")
    if save_mode == 'full':
        bigimagelist = os.listdir(big_image_path)
        bigimagelist.sort()
        biglabellist = os.listdir(big_label_path)
        biglabellist.sort()
    imagelist = os.listdir(image_path)
    imagelist.sort()    # 读取顺序异常 增加一个排序
    labellist = os.listdir(label_path)
    labellist.sort()    # 读取顺序异常 增加一个排序
    if test_mode == 'auto':
        # TODO: 自动计算不同overlap下每行瓦片数
        stride = global_image_size * (1 - overlay)
        times = int(5000 // stride) - int((global_image_size - stride) // stride) - 1
        sum_image = imagelist.__len__()
        batch_image = times
        itr = times
        sum_big = 5
        # 拼接后大小
        tr_size = int(stride * times)
    elif test_mode == 'manual':
        # 手动部分 1张
        sum_big = 1
        batch_image = global_test_image_num
        itr = global_test_itr
    else:
        print("test mode choose error")
        os._exit(1)

    if not os.path.exists(saved_results_path):
        os.makedirs(saved_results_path)
    #
    eval_p = {'Prescision': [], 'Recall': [],  'F_measure': [], 'IoU': [], 'Dice': []}
    eval_oc = {'Prescision': [], 'Recall': [],  'F_measure': [], 'IoU': [], 'Dice': []}
    eval_co = {'Prescision': [], 'Recall': [],  'F_measure': [], 'IoU': [], 'Dice': []}
    eval_crf = {'Prescision': [], 'Recall': [],  'F_measure': [], 'IoU': [], 'Dice': []}
    eval_crf_co = {'Prescision': [], 'Recall': [],  'F_measure': [], 'IoU': [], 'Dice': []}
    #
    sum_pixel = global_image_size * global_image_size
    out = 0
    time_start = time.time()
    #####
    #####
    for big_i in range(sum_big):
        mark = np.zeros([tr_size, tr_size, 2])
        for i in range(big_i * itr , (big_i + 1) * itr):
            testGene = testGenerator(imagelist, i * batch_image, image_path, batch_image)
            results = model.predict_generator(testGene, batch_image, verbose=2)
            # use_multiprocessing=True` is not supported on Windows
            tmp = 0
            # 原始概率图
            mark = merge_list_big(mark, results, i % itr, overlay, global_image_size).astype(np.float32)
            if save_mode == 'single':
                for k in range(i * batch_image, (i + 1) * batch_image):
                    print("mian" + str(k))
                #     #
                    image = np.array(imageio.imread(os.path.join(image_path, imagelist[k])))
                #     #
                    label = np.array(imageio.imread(os.path.join(label_path, labellist[k])))
                #
                #     label_01 = label / 255  # 归一化
                #
                #     # 全黑无效 && <5%无效
                #     if label_01.max() != 1:
                #         tmp += 1
                #         out += 1
                #         continue
                #     tmp_pixel = np.count_nonzero(label_01)
                #     if (tmp_pixel / sum_pixel) < 0.05:
                #         tmp += 1
                #         out += 1
                #         continue
                #     # pred score
                #     res = results[tmp][:, :, 1]
                #     tmp += 1
                #     # 阈值
                #     th = 0.5
                #     # 单CRF
                #     crf_res = crf.CRFs(image, res)
                #     # CRF + 闭 + 开
                #     crf_co_res = morph.morph(crf_res, operation='co', vary=True, th=th)
                #     #
                #     crf_res = vary(crf_res, th)
                #     #############################
                #     # 形态学开闭
                #     res_mor_oc = morph.morph(res, operation='oc', vary=True, th=th)
                #     #############################
                #     # 形态学闭开
                #     res_mor_co = morph.morph(res, operation='co', vary=True, th=th)
                #     #############################
                #     res = vary(res, th)
                #     # 全黑无效
                #     if res.max() != 1:
                #         out += 1
                #         continue
                #     # # Recall
                #     recall_orgin = eva.Recall(res, label_01)
                #     eval_p['Recall'].append(recall_orgin)
                #     recall_oc = eva.Recall(res_mor_oc, label_01)  # 结果不好
                #     eval_oc['Recall'].append(recall_oc)
                #     recall_co = eva.Recall(res_mor_co, label_01)
                #     eval_co['Recall'].append(recall_co)
                #     recall_crf = eva.Recall(crf_res, label_01)
                #     eval_crf['Recall'].append(recall_crf)
                #     recall_crf_co = eva.Recall(crf_co_res, label_01)
                #     eval_crf_co['Recall'].append(recall_crf_co)
                #     # # Percision
                #     prec_orgin = eva.Precision(res, label_01)
                #     eval_p['Prescision'].append(prec_orgin)
                #     prec_oc = eva.Precision(res_mor_oc, label_01)  # 结果不好
                #     eval_oc['Prescision'].append(prec_oc)
                #     prec_co = eva.Precision(res_mor_co, label_01)
                #     eval_co['Prescision'].append(prec_co)
                #     prec_crf = eva.Precision(crf_res, label_01)
                #     eval_crf['Prescision'].append(prec_crf)
                #     prec_crf_co = eva.Precision(crf_co_res, label_01)
                #     eval_crf_co['Prescision'].append(prec_crf_co)
                #     # # F-score
                #     F_orgin = eva.F_measure(recall_orgin, prec_orgin)
                #     eval_p['F_measure'].append(F_orgin)
                #     F_oc = eva.F_measure(recall_oc, prec_oc)  # 结果不好
                #     eval_oc['F_measure'].append(F_oc)
                #     F_co = eva.F_measure(recall_co, prec_co)
                #     eval_co['F_measure'].append(F_co)
                #     F_crf = eva.F_measure(recall_crf, prec_crf)
                #     eval_crf['F_measure'].append(F_crf)
                #     F_crf_co = eva.F_measure(recall_crf_co, prec_crf_co)
                #     eval_crf_co['F_measure'].append(F_crf_co)
                #     # IoU
                #     IoU_orgin = eva.mean_iou(res, label_01)
                #     eval_p['IoU'].append(IoU_orgin)
                #     IoU_oc = eva.mean_iou(res_mor_oc, label_01)  # 结果不好
                #     eval_oc['IoU'].append(IoU_oc)
                #     IoU_co = eva.mean_iou(res_mor_co, label_01)
                #     eval_co['IoU'].append(IoU_co)
                #     IoU_crf = eva.mean_iou(crf_res, label_01)
                #     eval_crf['IoU'].append(IoU_crf)
                #     IoU_crf_co = eva.mean_iou(crf_co_res, label_01)
                #     eval_crf_co['IoU'].append(IoU_crf_co)
                #     # dice
                #     dice_orgin = eva.dice(res, label_01)
                #     eval_p['Dice'].append(dice_orgin)
                #     dice_oc = eva.dice(res_mor_oc, label_01)  # 结果不好
                #     eval_oc['Dice'].append(dice_oc)
                #     dice_co = eva.dice(res_mor_co, label_01)
                #     eval_co['Dice'].append(dice_co)
                #     dice_crf = eva.dice(crf_res, label_01)
                #     eval_crf['Dice'].append(dice_crf)
                #     dice_crf_co = eva.dice(crf_co_res, label_01)
                #     eval_crf_co['Dice'].append(dice_crf_co)
                #
                #     print(imagelist[k])
                #     print("-" * 10)
                #
                #     # Out
                    imageio.imwrite(os.path.join(saved_results_path, "%d_image.jpg" % k), image)
                    imageio.imwrite(os.path.join(saved_results_path, "%d_gt.jpg" % k), label)
                    # imageio.imwrite(os.path.join(saved_results_path, "%d_2_predict.jpg" % k), res)
                    # imageio.imwrite(os.path.join(saved_results_path, "%d_2_mor_oc_predict.jpg" % k), res_mor_oc) # 结果不好
                    # imageio.imwrite(os.path.join(saved_results_path, "%d_2_mor_co_predict.jpg" % k), res_mor_co)
                    # imageio.imwrite(os.path.join(saved_results_path, "%d_2_crf_predict.jpg" % k), crf_res)
                    # imageio.imwrite(os.path.join(saved_results_path, "%d_2_crf_co_predict.jpg" % k), crf_co_res)

        if save_mode == 'full':
            print(bigimagelist[big_i])
            image = np.array(imageio.imread(os.path.join(big_image_path, bigimagelist[big_i])))[:tr_size, :tr_size, :]
            label = np.array(imageio.imread(os.path.join(big_label_path, biglabellist[big_i])))[:tr_size, :tr_size]
            label_01 = label / 255  # 归一化
            th = 0.6
            mark = mark[:, :, 1]
            # 单CRF
            crf_res = crf.CRFs(image, mark, tr_size)
            # CRF + 闭开
            crf_co_res = morph.morph(crf_res, operation='co', vary=True, th=th)
            #############################
            # 形态学开闭
            res_mor_oc = morph.morph(mark, operation='oc', vary=True, th=th)    # 结果不好
            #############################
            # 形态学闭开
            res_mor_co = morph.morph(mark, operation='co', vary=True, th=th)
            #############################
            res = vary(mark, th)
            # # Recall - Precision - F
            recall_orgin, prec_orgin, F_orgin = eva.RecPreF(res, label_01, tr_size)
            eval_p['Recall'].append(recall_orgin)
            eval_p['Prescision'].append(prec_orgin)
            eval_p['F_measure'].append(F_orgin)
            #
            # recall_oc, prec_oc, F_oc = eva.RecPreF(res_mor_oc, label_01, tr_size)  # 结果不好
            # eval_oc['Recall'].append(recall_oc)
            # eval_oc['Prescision'].append(prec_oc)
            # eval_oc['F_measure'].append(F_oc)
            # #
            # recall_co, prec_co, F_co = eva.RecPreF(res_mor_co, label_01, tr_size)
            # eval_co['Recall'].append(recall_co)
            # eval_co['Prescision'].append(prec_co)
            # eval_co['F_measure'].append(F_co)
            #
            recall_crf, prec_crf, F_crf = eva.RecPreF(crf_res, label_01, tr_size)
            eval_crf['Recall'].append(recall_crf)
            eval_crf['Prescision'].append(prec_crf)
            eval_crf['F_measure'].append(F_crf)
            #
            recall_crf_co, prec_crf_co, F_crf_co = eva.RecPreF(crf_co_res, label_01, tr_size)
            eval_crf_co['Recall'].append(recall_crf_co)
            eval_crf_co['Prescision'].append(prec_crf_co)
            eval_crf_co['F_measure'].append(F_crf_co)
            # IoU - Dice
            IoU_orgin, dice_orgin = eva.IouDice(res, label_01)
            eval_p['IoU'].append(IoU_orgin)
            eval_p['Dice'].append(dice_orgin)
            #
            # IoU_oc, dice_oc = eva.IouDice(res_mor_oc, label_01)  # 结果不好
            # eval_oc['IoU'].append(IoU_oc)
            # eval_oc['Dice'].append(dice_oc)
            # #
            # IoU_co, dice_co = eva.IouDice(res_mor_co, label_01)
            # eval_co['IoU'].append(IoU_co)
            # eval_co['Dice'].append(dice_co)
            #
            IoU_crf, dice_crf = eva.IouDice(crf_res, label_01)
            eval_crf['IoU'].append(IoU_crf)
            eval_crf['Dice'].append(dice_crf)
            #
            IoU_crf_co, dice_crf_co = eva.IouDice(crf_co_res, label_01)
            eval_crf_co['IoU'].append(IoU_crf_co)
            eval_crf_co['Dice'].append(dice_crf_co)

            print(bigimagelist[big_i])
            print("-" * 10)

            imageio.imwrite(os.path.join(saved_results_path, "big_predict.jpg"), mark)
            imageio.imwrite(os.path.join(saved_results_path, "%s_image.jpg" % bigimagelist[big_i]), image)
            imageio.imwrite(os.path.join(saved_results_path, "%s_gt.jpg" % bigimagelist[big_i]), label)
            imageio.imwrite(os.path.join(saved_results_path, "%s_2_predict.jpg" % bigimagelist[big_i]), res)
            # imageio.imwrite(os.path.join(saved_results_path, "%s_2_mor_oc_predict.jpg" % bigimagelist[big_i]), res_mor_oc) # 结果不好
            # imageio.imwrite(os.path.join(saved_results_path, "%s_2_mor_co_predict.jpg" % bigimagelist[big_i]), res_mor_co)
            imageio.imwrite(os.path.join(saved_results_path, "%s_2_crf_predict.jpg" % bigimagelist[big_i]), crf_res)
            imageio.imwrite(os.path.join(saved_results_path, "%s_2_crf_co_predict.jpg" % bigimagelist[big_i]), crf_co_res)

    #均值方差
    print("origin")
    eva.mean_variance(eval_p)
    # print("oc")
    # eva.mean_variance(eval_oc)
    # print("co")
    # eva.mean_variance(eval_co)
    print("crf")
    eva.mean_variance(eval_crf)
    print("crf_co")
    eva.mean_variance(eval_crf_co)
    #
    import GUI.result as ts
    ts.box_show(eval_p, saved_results_path, "origin")
    # ts.box_show(eval_oc, saved_results_path, "open-close")
    # ts.box_show(eval_co, saved_results_path, "close-open")
    ts.box_show(eval_crf, saved_results_path, "crf")
    ts.box_show(eval_crf_co, saved_results_path, "crf_co")

    #finish
    delta_time = time.time() - time_start
    print("delta_time" + str(delta_time))
    print("sum_image==" + str(batch_image * itr - out))
