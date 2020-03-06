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

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=20, srgb=20, rgbim=rbg_img, compat=10)  # pairwise energy

    Q = d.inference(5)  # inference 5 times
    Q = np.argmax(np.array(Q), axis=0).reshape((256, 256)).astype(np.float32)
    
    return Q