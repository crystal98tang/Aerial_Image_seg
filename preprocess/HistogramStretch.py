import numpy as np
from skimage import exposure,data
import imageio
import matplotlib.pyplot as plt

img = np.array(imageio.imread("../testImage/0.tif"))
gt = np.array(imageio.imread("../testImage/0_gt.tif"))

# plt.draw(img)
# plt.pause(4)# 间隔的秒数： 4s
# plt.draw(gt)
# plt.pause(4)# 间隔的秒数： 4s
# #
hist1=np.histogram(img, bins=2)   #用numpy包计算直方图
print(hist1)
#
plt.figure("hist")
arr=img.flatten() #将二维数组序列化成一维数组。是按行序列，如mat=[[1 2 3 4 5 6]] 经过 mat.flatten()后，就变成了 mat=[1 2 3 4 5 6]
n, bins, patches = plt.hist(arr, bins=256, density=1,edgecolor='None',facecolor='red')
plt.draw()
plt.pause(1)# 间隔的秒数： 4s
#
ar=img[:,:,0].flatten()
plt.hist(ar, bins=256, density=1,facecolor='r',edgecolor='r')
ag=img[:,:,1].flatten()
plt.hist(ag, bins=256, density=1, facecolor='g',edgecolor='g')
ab=img[:,:,2].flatten()
plt.hist(ab, bins=256, density=1, facecolor='b',edgecolor='b')
plt.draw()
plt.pause(1)# 间隔的秒数： 4s