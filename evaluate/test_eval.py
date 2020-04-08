import evaluate.utils as eval
import numpy as np
a = np.array([[1,0,1],[0,1,0],[1,0,0]])
b = np.array([[1,0,0],[0,1,0],[0,0,1]])

eval.mean_iou(a,b)

eval.dice(a,b)

eval.dc(a,b)