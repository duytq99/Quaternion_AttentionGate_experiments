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
from   core_qnn.quaternion_ops import *

class Attention(nn.Module):
    def __init__(self,ch_ing,ch_inx,ch_out=512,kernel_size=1):
        super(Attention,self).__init__()
        self.convg = nn.Conv2d(ch_ing, ch_out, kernel_size=kernel_size, stride=1,padding=0,bias=False)
        self.convx = nn.Conv2d(ch_inx, ch_out, kernel_size=kernel_size, stride=1,padding=0,bias=False)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(ch_out, 1, kernel_size=kernel_size, stride=1,padding=0,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,g,x):
        g1 = self.convg(g)
        
        x1 = self.convx(x)

        net = g1 + x1
        net = self.relu(net)
        net = self.conv(net)
        net = self.sigmoid(net)
        net = net*x
        return net

class Q_Attention(nn.Module):
    def __init__(self,ch_ing,ch_inx,ch_out=512,kernel_size=1):
        super(Q_Attention,self).__init__()
        self.convg = QuaternionConv(ch_ing, ch_out, kernel_size=kernel_size, stride=1,padding=0,bias=False)
        self.convx = QuaternionConv(ch_inx, ch_out, kernel_size=kernel_size, stride=1,padding=0,bias=False)
        self.relu = nn.ReLU()
        self.conv = QuaternionConv(ch_out, 4, kernel_size=kernel_size, stride=1,padding=0,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,g,x):
        g1 = self.convg(g)
        
        x1 = self.convx(x)

        net = g1 + x1
        net = self.relu(net)
        net = self.conv(net)
        net = get_modulus(net, True)
        net = self.sigmoid(net)
        net = net*x
        return net

def Upsample(tensor, rate=2):
    shape = list(tensor.size())
    return nn.functional.interpolate(tensor,size=[shape[2]*rate,shape[3]*rate])

class A_QCAE(nn.Module):

    def __init__(self):
        super(A_QCAE, self).__init__()

        self.act        = nn.ReLU()
        self.output_act = nn.ReLU()

        # ENCODER
        self.e1 = QuaternionConv(4, 32, kernel_size=3, stride=2, padding=1)
        self.e2 = QuaternionConv(32, 40, kernel_size=3, stride=2, padding=1)
        self.att = Attention(32,32,32)
        # DECODER
        self.d5 = QuaternionTransposeConv(40, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d6 = QuaternionTransposeConv(64, 4, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):

        e1 = self.act(self.e1(x)) #
        e2 = self.act(self.e2(e1))
        
        d5 = self.act(self.d5(e2)) #

        attention = self.att(e1,d5)

        attention_r = get_r(attention)
        attention_i = get_i(attention)
        attention_j = get_j(attention)
        attention_k = get_k(attention)

        d5_r = get_r(d5)
        d5_i = get_i(d5)
        d5_j = get_j(d5)
        d5_k = get_k(d5)

        cat_r = torch.cat((attention_r,d5_r), 1)
        cat_i = torch.cat((attention_i,d5_i), 1)
        cat_j = torch.cat((attention_j,d5_j), 1)
        cat_k = torch.cat((attention_k,d5_k), 1)

        cat = torch.cat((cat_r,cat_i,cat_j,cat_k), 1)

        d6 = self.d6(cat)
        out = self.output_act(d6)

        return out
    def name(self):
        return "A_QCAE"

class QCAE(nn.Module):

    def __init__(self):
        super(QCAE, self).__init__()

        self.act        = nn.ReLU()
        self.output_act = nn.ReLU()

        # ENCODER
        self.e1 = QuaternionConv(4, 32, kernel_size=3, stride=2, padding=1)
        self.e2 = QuaternionConv(32, 40, kernel_size=3, stride=2, padding=1)
        # self.att = Q_Attention(32,32,32)
        # DECODER
        self.d5 = QuaternionTransposeConv(40, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d6 = QuaternionTransposeConv(64, 4, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):

        e1 = self.act(self.e1(x)) #
        e2 = self.act(self.e2(e1))
        
        d5 = self.act(self.d5(e2)) #

        
        cat = torch.cat((d5,e1), 1)

        d6 = self.d6(cat)
        out = self.output_act(d6)

        return out

    def name(self):
        return "QCAE"

class QA_QCAE(nn.Module):

    def __init__(self):
        super(QA_QCAE, self).__init__()

        self.act        = nn.ReLU()
        self.output_act = nn.ReLU()

        # ENCODER
        self.e1 = QuaternionConv(4, 32, kernel_size=3, stride=2, padding=1)
        self.e2 = QuaternionConv(32, 40, kernel_size=3, stride=2, padding=1)
        self.att = Q_Attention(32,32,32)
        # DECODER
        self.d5 = QuaternionTransposeConv(40, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d6 = QuaternionTransposeConv(64, 4, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):

        e1 = self.act(self.e1(x)) #
        e2 = self.act(self.e2(e1))
        
        d5 = self.act(self.d5(e2)) #

        attention = self.att(e1,d5)

        attention_r = get_r(attention)
        attention_i = get_i(attention)
        attention_j = get_j(attention)
        attention_k = get_k(attention)

        d5_r = get_r(d5)
        d5_i = get_i(d5)
        d5_j = get_j(d5)
        d5_k = get_k(d5)

        cat_r = torch.cat((attention_r,d5_r), 1)
        cat_i = torch.cat((attention_i,d5_i), 1)
        cat_j = torch.cat((attention_j,d5_j), 1)
        cat_k = torch.cat((attention_k,d5_k), 1)

        cat = torch.cat((cat_r,cat_i,cat_j,cat_k), 1)

        d6 = self.d6(cat)
        out = self.output_act(d6)

        return out

    def name(self):
        return "QA_QCAE"

if __name__ == "__main__":
    
    model = QCAE()
    summary(model, (4,512,512),col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])