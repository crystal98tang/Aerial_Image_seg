import cv2


# morphological opening-and-closing operation
def morph(predict, pred_max=1, operation='oc', vary=True, th=0.5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_itr = 3
    if pred_max == 255:
        predict /= 255
    #
    res = None
    if operation == 'oc':
        res = cv2.morphologyEx(predict, cv2.MORPH_OPEN, kernel, iterations=k_itr)
        res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=k_itr)
    elif operation == 'co':
        res = cv2.morphologyEx(predict, cv2.MORPH_CLOSE, kernel, iterations=3)
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel, iterations=3)
    else:
        res = -1
    #
    if vary:
        res[res > th] = 1
        res[res < th] = 0
    #
    return res
