import numpy as np
import pydensecrf.densecrf as dcrf


def CRFs(original_image, predicted_image):

    rbg_img = original_image
    pred_score = np.expand_dims(predicted_image, 0)
    pred_score = np.append(1 - pred_score, pred_score, axis=0)
    # sigm_score: 经过sigmoid的score map, size=[H, W]
    d = dcrf.DenseCRF2D(256, 256, 2)  # 2 classes, width first then height
    U = -np.log(pred_score)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    rbg_img = np.ascontiguousarray(rbg_img)
    
    d.setUnaryEnergy(U)  # add unary

    # 增加了与颜色无关的术语，只是位置-----会惩罚空间上孤立的小块分割,即强制执行空间上更一致的分割
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # 增加了颜色相关术语，即特征是(x,y,r,g,b)-----使用局部颜色特征来细化它们
    d.addPairwiseBilateral(sxy=(40, 40), srgb=(13, 13, 13), rgbim=rbg_img, compat=5,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)  # pairwise energy

    Q = d.inference(5)  # inference 5 times
    Q = np.argmax(np.array(Q), axis=0).reshape((256, 256)).astype(np.float32)
    
    return Q