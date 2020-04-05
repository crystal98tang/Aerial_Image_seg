import evaluate.utils as eva
import imageio

gt_path = ""
pred_path = ""

gt = imageio.imread(gt_path)
pred = imageio.imread(pred_path)

prec = eva.Precision(pred,gt)
recall = eva.Recall(pred,gt)
F = eva.F_measure(recall, prec)
IoU = eva.mean_iou(pred,gt)
dice = eva.dice(pred,gt)

print("prec:%.2f\trecall:%.2f\tF:%.2f\tIoU:%.f\t" + )