import torch
import torch.nn as nn
from   torch.nn import Parameter
from   torch.nn import functional as F
from torchsummary import summary
import torch.optim
from   torch.autograd import Variable
from   torch import autograd
import numpy as np

from   core_qnn.quaternion_layers import QuaternionLinear, QuaternionTransposeConv, QuaternionMaxAmpPool2d, QuaternionConv

class attention(nn.Module):
    def __init__(self,ch_ing,ch_inx,ch_out=512,kernel_size=1):
        super(attention,self).__init__()
        self.convg = nn.Conv2d(ch_ing, ch_out, kernel_size=kernel_size, stride=1,padding=0,bias=False)
        self.convx = nn.Conv2d(ch_inx, ch_out, kernel_size=kernel_size, stride=1,padding=0,bias=False)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(ch_out, 1, kernel_size=kernel_size, stride=1,padding=0,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,g,x):
        g1 = self.convg(g)
        
        x1 = self.convx(x)
        print(g1.shape, x1.shape)
        net = g1 + x1
        net = self.relu(net)
        net = self.conv(net)
        net = self.sigmoid(net)
        net = net*x
        return net

def Upsample(tensor, rate=2):
    shape = list(tensor.size())
    return nn.functional.interpolate(tensor,size=[shape[2]*rate,shape[3]*rate])

class A_QCAE(nn.Module):

    def __init__(self):
        super(A_QCAE, self).__init__()

        # self.act        = nn.Hardtanh()
        # self.output_act = nn.Hardtanh()
        self.act        = nn.ReLU()

        # ENCODER
        self.conv1      = QuaternionConv(4, 32, kernel_size=3, stride=1, padding=1)        
        self.conv2      = QuaternionConv(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1      = QuaternionMaxAmpPool2d(2, 2)  
        self.conv3      = QuaternionConv(64, 64, kernel_size=3, stride=1, padding=1)
        
        # DECODER      
        self.upconv1    = QuaternionTransposeConv(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention1 = attention(32,64,32)
        self.conv4      = QuaternionConv(96,64, kernel_size=3, stride=1, padding=1)
        self.conv5      = nn.Conv2d(64,3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):

        layer1 = self.act(self.conv1(x))
        layer2 = self.pool1(self.act(self.conv2(layer1)))
        layer3 = self.act(self.conv3(layer2))
        layer4_1 = self.act(self.upconv1(layer3))
        layer4_2 = self.attention1(layer1,Upsample(layer3))
        layer4 = torch.cat((layer4_1, layer4_2), axis=1)
        layer5 = self.act(self.conv4(layer4))
        output = self.conv5(layer5)

        return output

    def name(self):
        return "QCAE"

if __name__ == "__main__":
    
    model = A_QCAE()
    summary(model, (4,512,512))