import torch 
import torch.nn as nn
import torchvision
from torchsummary import summary
import numpy as np 
import matplotlib.pyplot as plt 
import imageio
import time
from quaternion_layers import QuaternionLinear
from model import CIFAR10ConvNet, CIFAR10QConvNet, CIFAR10QConvNetBN, CIFAR10ConvNetBN, CIFAR10QConvNetBN_NormalPool, CIFAR10QConvNet_NormalPool, resnet50, resnet50_quat

def create_model(quaternion):
    if quaternion:
      model = resnet50_quat(num_classes=102, pretrained=False)
      num_filters = model.fc.in_features
      print(num_filters)
      model.fc = nn.Sequential(QuaternionLinear(num_filters*4, 512*4),
                               nn.ReLU(),
                               QuaternionLinear(512*4, 102*4))
    else:
      model = resnet50(num_classes=102, pretrained=False)
      num_filters = model.fc.in_features
      print(num_filters)
      model.fc = nn.Sequential(nn.Linear(num_filters, 512),
                               nn.ReLU(),
                               nn.Linear(512, 102))

    model.to(device)
    return model

if __name__ == "__main__":
    
    epochs = 50
    init_learning_rate = 0.01
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    batch_size_training = 128
    batch_size_validation = 128

    model = create_model(quaternion=True)
    model.to(device)
    # summary(model = model, input_data=(4,256,256), col_names =["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], col_width = 30, verbose = 1)
    summary(model = model, input_data=(4,256,256), col_width = 30, verbose = 1)
