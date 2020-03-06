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


if run_mode == "test":
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
    eval_crf = {'Recall': [], 'Prescision': [], 'F_measure': [], 'IoU': [], 'Dice': []}
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
            crf_res = crf.CRFs(image, res)
            res = vary(res, th)
            # 全黑无效
            if res.max() != 1:
                continue
            # # Recall
            recall_orgin = eva.Recall(res, label_01)
            eval_p['Recall'].append(recall_orgin)
            recall_crf = eva.Recall(res, label_01)
            eval_crf['Recall'].append(recall_crf)
            # # Percision
            prec_orgin = eva.Precision(res, label_01)
            eval_p['Prescision'].append(prec_orgin)
            recall_crf = eva.Recall(res, label_01)
            eval_crf['Recall'].append(recall_crf)
            # # F-score
            F_orgin = eva.F_measure(recall_orgin, prec_orgin)
            eval_p['F_measure'].append(F_orgin)
            recall_crf = eva.Recall(res, label_01)
            eval_crf['Recall'].append(recall_crf)
            # IoU
            IoU_orgin = eva.mean_iou(res, label_01)
            eval_p['IoU'].append(IoU_orgin)
            recall_crf = eva.Recall(res, label_01)
            eval_crf['Recall'].append(recall_crf)
            # dice
            dice_orgin = eva.dice(res, label_01)
            eval_p['Dice'].append(dice_orgin)
            recall_crf = eva.Recall(res, label_01)
            eval_crf['Recall'].append(recall_crf)

            print(imagelist[k])
            print("-" * 10)

            # Out
            imageio.imwrite(os.path.join(saved_results_path, "%d_image.tif" % k), image)
            imageio.imwrite(os.path.join(saved_results_path, "%d_gt.tif" % k), label)
            imageio.imwrite(os.path.join(saved_results_path, "%d_2_predict.tif" % k), res)
            # imageio.imwrite(os.path.join(saved_results_path, "%d_2_mor_oc_predict.tif" % k), res_mor_oc) # 结果不好
            # imageio.imwrite(os.path.join(saved_results_path, "%d_2_mor_co_predict.tif" % k), res_mor_co)
            imageio.imwrite(os.path.join(saved_results_path, "%d_2_crf_predict.tif" % k), crf_res)

    import GUI.result as ts

    ts.box_show(eval_p)
    ts.box_show(eval_oc)
    ts.box_show(eval_co)