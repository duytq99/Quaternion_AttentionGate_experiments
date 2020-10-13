import torch
import torch.nn as nn
from   torch.nn import Parameter
from   torch.nn import functional as F
from torchsummary import summary
import torch.optim
from   torch.autograd import Variable
from   torch import autograd
import numpy as np

from   core_qnn.quaternion_layers import QuaternionLinear, QuaternionTransposeConv, QuaternionConv


class QAE(nn.Module):

    def __init__(self):
        super(QAE, self).__init__()

        self.act        = nn.ReLU()
        self.output_act = nn.Sigmoid()

        # ENCODER
        self.e1 = QuaternionLinear(65536, 4096)

        # DECODER
        self.d1 = QuaternionLinear(4096, 65536)


    def forward(self, x):

        e1 = self.act(self.e1(x))
        d1 = self.d1(e1)

        out = self.output_act(d1)
        return out

    def name(self):
        return "QAE"

class QCAE(nn.Module):

    def __init__(self):
        super(QCAE, self).__init__()

        self.act        = nn.Hardtanh()
        self.output_act = nn.Hardtanh()

        # ENCODER
        self.e1 = QuaternionConv(4, 32, kernel_size=3, stride=2, padding=1)
        self.e2 = QuaternionConv(32, 40, kernel_size=3, stride=2, padding=1)

        # DECODER
        self.d5 = QuaternionTransposeConv(40, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d6 = QuaternionTransposeConv(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):

        e1 = self.act(self.e1(x))
        e2 = self.act(self.e2(e1))

        d5 = self.act(self.d5(e2))
        d6 = self.d6(d5)
        out = self.output_act(d6)

        return out

    def name(self):
        return "QCAE"


model = QCAE()
summary(model, (4,512,512))