import torch
from   torch                import nn
from   attention_model import A_QCAE


import os
from   imageio import imread,imwrite
import numpy   as np
import sys

from icassp_2019.utils.psnr_ssim import psnr

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


# MODEL         = str(sys.argv[1])
MODEL         = 'QCAE'
CUDA          = False
NUM_EPOCHS    = 1000
LEARNING_RATE = 0.0005

if MODEL == 'QCAE':
    net  = A_QCAE()
# else:
#     net  = CAE()

if CUDA:
    net = net.cuda()

#
# LOAD PICTURE
#

# os.system('rm -rf data/save_image*')
# if not os.path.isdir('icassp_2019/out'):
#     os.system('mkdir icassp_2019/out')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

train = rgb2gray(imread('icassp_2019/data/kodak/kodim05.png'))
train_orig = rgb2gray(imread('icassp_2019/data/kodak/kodim05.png'))
imwrite("icassp_2019/out/save_image_gray_training.png", train)
imwrite("icassp_2019/out/girl_image_gray_training.png", rgb2gray(imread('icassp_2019/data/kodak/kodim04.png')))
imwrite("icassp_2019/out/parrot_image_gray_training.png",rgb2gray(imread('icassp_2019/data/kodak/kodim23.png')))
train = train / 255

test = imread('icassp_2019/data/kodak/kodim04.png')
test = test / 255

nb_param = sum(p.numel() for p in net.parameters() if p.requires_grad)

print("QCAE & CAE Color images - Titouan Parcollet - LIA, ORKIS")
print("Model Info --------------------")
print("Number of trainable parameters : "+str(nb_param))

if MODEL == 'QCAE':
    npad  = ((0, 0), (0, 0), (1, 0))
    train = np.pad(train, pad_width=npad, mode='constant', constant_values=0)
    train = np.transpose(train, (2,0,1))
    train = np.reshape(train, (1, train.shape[0], train.shape[1], train.shape[2]))
    
    # train_orig = np.pad(train_orig, pad_width=npad, mode='constant', constant_values=0)
    train_orig = np.transpose(train_orig, (2,0,1))
    train_orig = np.reshape(train_orig, (1, train_orig.shape[0], train_orig.shape[1], train_orig.shape[2]))

    # print(train.shape)

    test = np.pad(test, pad_width=npad, mode='constant', constant_values=0)
    test = np.transpose(test, (2,0,1))
    test = np.reshape(test, (1, test.shape[0], test.shape[1], test.shape[2]))

else:
    train = np.transpose(train, (2,0,1))
    train = np.reshape(train, (1, train.shape[0], train.shape[1], train.shape[2]))

    test  = np.transpose(test, (2,0,1))
    test  = np.reshape(test, (1, test.shape[0], test.shape[1], test.shape[2]))

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
    
    print("epoch "+str(epoch)+", loss_train "+str(loss.cpu().item())+"psnr "+str(psnr_print))
    # print("psnr ", psnr_print)
    
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
        imwrite("icassp_2019/out/save_image"+str(epoch)+".png", out)
    