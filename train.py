import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from __future__ import division
from __future__ import print_function
import os, time, scipy.io
import tensorflow as tf
import numpy as np
import tifffile
import pdb
import glob
# from tensorflow.contrib.layers.python.layers import initializers

# from tensorflow.python import pywrap_tensorflow

# import model

input_dir = './dataset/short/'
gt_dir = './dataset/long/'
checkpoint_dir = './checkpoint/new_at_unet2/'
result_dir = './result2/'

#get train and test IDs
train_fns = glob.glob(gt_dir + '0*.tiff')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))
test_fns = glob.glob(gt_dir + '/1*.tiff')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))

ps = 512 #patch size for training

save_freq = 10

DEBUG = 0
if DEBUG == 1:
    save_freq = 1
    train_ids = train_ids[0:5]
    test_ids = test_ids[0:5]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gt_images=[None]*6000
input_images = {}
input_images['300'] = [None]*len(train_ids)
input_images['250'] = [None]*len(train_ids)
input_images['100'] = [None]*len(train_ids)

g_loss = np.zeros((5000,1))

allfolders = glob.glob('./result1/*0')
lastepoch = 0
for folder in allfolders:
    # lastepoch = np.maximum(lastepoch, int(folder[-4:]))
    lastepoch = 0

learning_rate = 1e-4
net = AUNET(n_classes=3).to(device=device)
G_loss = nn.L1Loss(reduction='mean').to(device=device)
# lastepoch = 2
for epoch in range(lastepoch,4001):
    if os.path.isdir("result1/%04d"%epoch):
        continue    
    cnt=0
    if epoch > 2000:
        learning_rate = 1e-5
    G_opt = torch.optim.Adam(net.parameters(),lr=learning_rate)
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.tiff'%train_id)
        in_path = in_files[np.random.randint(0,len(in_files))]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.tiff'%train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-6])
        gt_exposure =  float(gt_fn[9:-6])
        ratio = min(gt_exposure/in_exposure,300)
        
        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:     
            in_img = tifffile.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(in_img,axis = 0)

        gt_img = tifffile.imread(gt_path)
        gt_img = np.expand_dims(gt_img,axis = 0)

        #crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0,W-ps)
        yy = np.random.randint(0,H-ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:,yy:yy+ps,xx:xx+ps,:]
        gt_patch =gt_img[:,yy:yy+ps,xx:xx+ps,:]
        input_patch = np.float32(input_patch)/65535.0*ratio
        gt_patch = np.float32(gt_patch)/65535.0

        if np.random.randint(2,size=1)[0] == 1:  # random flip 
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1: 
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose 
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))
        
        input_patch = torch.from_numpy(np.minimum(input_patch,1.0)).type(torch.FloatTensor).permute(0, 3, 1, 2).cuda()
        gt_patch = torch.from_numpy(np.minimum(gt_patch,1.0)).type(torch.FloatTensor).permute(0, 3, 1, 2).cuda()
        in_image = input_patch.to(device=device)
        gt_image = gt_patch.to(device=device)
        output = net(input_patch)
        loss = G_loss(output, gt_image)

        G_opt.zero_grad()
        loss.backward()
        G_opt.step()

        # _,G_current,output=sess.run([G_opt, G_loss, out_image],feed_dict={in_image:input_patch,gt_image:gt_patch,lr:learning_rate})
        output = torch.clamp(output, 0.0, 1.0)
        g_loss[ind]=loss.detach().cpu().numpy()

        print("%d %d Loss=%f Time=%f"%(epoch,cnt,g_loss[ind],time.time()-st))
        if epoch==0:
            if not os.path.isdir(result_dir + '%04d'%epoch):
                os.makedirs(result_dir + '%04d'%epoch)
            # temp = np.concatenate((gt_patch[0,:,:,:],output[0,:,:,:]),axis=1)
            # scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch,train_id,ratio))
    # saver.save(sess, checkpoint_dir + 'model.ckpt')