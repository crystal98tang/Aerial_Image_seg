import numpy as np
import pydensecrf.densecrf as dcrf


def CRFs(original_image, predicted_image):
    """
    original_image_path  原始图像路径
    predicted_image_path  之前用自己的模型预测的图像路径
    CRF_image_path  即将进行CRF后处理得到的结果图像保存路径
    """
    rbg_img = original_image
    pred_score = predicted_image

    # sigm_score: 经过sigmoid的score map, size=[H, W]
    # 需要特别注意,这里先w后h
    d = densecrf.DenseCRF2D(512, 512, 2)  # 2 classes, width first then height
    U = np.expand_dims(-np.log(pred_score), axis=0)  # [1, H, W], foreground
    U_ = np.expand_dims(-np.log(1 - pred_score), axis=0)  # [1, H, W], background
    unary = np.concatenate((U_, U), axis=0)
    unary = unary.reshape((2, -1))  # flatten, [2, HW], define unary
    d.setUnaryEnergy(unary)  # add unary

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=rbg_img, compat=10)  # pairwise energy

    Q = d.inference(5)  # inference 5 times
    pred_raw_dcrf = np.argmax(Q, axis=0).reshape((512, 512)).astype(np.float32)

    return pred_raw_dcrf