##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import numpy as np
import math
from scipy           import misc
import sys
from skimage.measure import compare_ssim as ssim
import torch

#
# Give the PSNR and SSIM between two given images
#


# original = misc.imread(sys.argv[1])
# contrast = misc.imread(sys.argv[2])

def psnr(img1, img2):
    # print(img1.shape, img2.shape)
    mseImage = (img1 - img2)**2
    
    mse = torch.mean(mseImage)

    
    # if mse == 0:
    #     return 100
    
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# p = psnr(original,contrast)
# s = ssim(original, contrast, multichannel=True)

# print("PSNR : "+str(p))
# print("SSIM : "+str(s))
