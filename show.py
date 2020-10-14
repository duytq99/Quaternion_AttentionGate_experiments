import cv2
import numpy as np

im = cv2.imread("E:\Project_DL\Quaternion_AttentionGate_experiments\icassp_2019\out\save_image100.png");
# print(im.shape)
# cv2.imshow('a',im)
# cv2.waitKey(0)

print(np.mean(im))