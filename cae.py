import torch
from   torch                import nn
from   attention_model import A_QCAE, QCAE, QA_QCAE

import os
from   imageio import imread,imwrite
import numpy   as np
import sys

import pytorch_ssim
import torch
from torch.autograd import Variable

from icassp_2019.utils.psnr_ssim import psnr

from torchsummary import summary as summary

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


MODEL         = str(sys.argv[1])
# MODEL         = 'QCAE'
CUDA          = False
NUM_EPOCHS    = 2000
LEARNING_RATE = 0.0005

if MODEL == 'QCAE':
    net  = QCAE()
    if not os.path.isdir('icassp_2019/out_QCAE'):
        os.mkdir('icassp_2019/out_QCAE')
    output_dir = 'icassp_2019/out_QCAE'
elif MODEL == 'A_QCAE':
    net  = A_QCAE()
    if not os.path.isdir('icassp_2019/out_A_QCAE'):
        os.mkdir('icassp_2019/out_A_QCAE')
    output_dir = 'icassp_2019/out_A_QCAE'
else:
    net = QA_QCAE()
    if not os.path.isdir('icassp_2019/out_QA_QCAE'):
        os.mkdir('icassp_2019/out_QA_QCAE')
    output_dir = 'icassp_2019/out_QA_QCAE'
    
if CUDA:
    net = net.cuda()

#
# LOAD PICTURE
#
"""
os.system('rm -rf data/out/save_image*')
if not os.path.isdir('icassp_2019/out'):
    os.system('mkdir icassp_2019/out')
"""
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

train = rgb2gray(imread('icassp_2019/data/kodak/kodim05.png'))
train_orig = rgb2gray(imread('icassp_2019/data/kodak/kodim05.png'))
imwrite(os.path.join(output_dir,"save_image_gray_training.png"), train)
imwrite(os.path.join(output_dir,"girl_image_gray_training.png"), rgb2gray(imread('icassp_2019/data/kodak/kodim04.png')))
imwrite(os.path.join(output_dir,"parrot_image_gray_training.png"),rgb2gray(imread('icassp_2019/data/kodak/kodim23.png')))
train = train / 255

test = imread('icassp_2019/data/kodak/kodim23.png')
test = test / 255

nb_param = sum(p.numel() for p in net.parameters() if p.requires_grad)

print("QCAE & CAE Color images - Titouan Parcollet - LIA, ORKIS")
print("Model Info --------------------")
print("Number of trainable parameters : "+str(nb_param))
summary(net, (4,512,512),col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])

if MODEL == 'QCAE' or 'A_QCAE' or 'QA_QCAE':
    npad  = ((0, 0), (0, 0), (1, 0))
    train = np.pad(train, pad_width=npad, mode='constant', constant_values=0)
    train = np.transpose(train, (2,0,1))
    train = np.reshape(train, (1, train.shape[0], train.shape[1], train.shape[2]))

    test = np.pad(test, pad_width=npad, mode='constant', constant_values=0)
    test = np.transpose(test, (2,0,1))
    test = np.reshape(test, (1, test.shape[0], test.shape[1], test.shape[2]))

if CUDA:
    train = torch.from_numpy(train).float().cuda()
    test  = torch.from_numpy(test).float().cuda()
else:
    train = torch.from_numpy(train).float()
    train_orig = torch.from_numpy(train_orig).float()
    test  = torch.from_numpy(test).float()

for epoch in range(NUM_EPOCHS):
    # print(train.shape)    

    output = net(train)
    loss   = criterion(output, train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    psnr_print = psnr(output[:,1:,:,:],train[:,1:,:,:])
    ssim_print = pytorch_ssim.ssim(train[:,1:,:,:], output[:,1:,:,:]).item()
    
    print("epoch {} loss_train {} psnr {} ssim {}".format(epoch, loss.cpu().item(), psnr_print, ssim_print))
    
    if (epoch %100) == 0:

        output = net(test)
        out    = output.cpu().data.numpy()
        if MODEL == 'QCAE':
            out = np.transpose(out, (0,2,3,1))[:,:,:,1:]
            out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))
        else:
            out = np.transpose(out, (0,2,3,1))
            out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))

        # print(out.shape)
        imwrite(os.path.join(output_dir,"save_image"+str(epoch)+".png"), out)
    