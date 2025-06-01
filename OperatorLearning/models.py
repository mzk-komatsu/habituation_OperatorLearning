"""
This file includes portions of code from fourier_neural_operator (https://github.com/wesley-stone/fourier_neural_operator/) by Zongyi Li,
which is licensed under the MIT License. 
The copy of repository "fourier_neural_operator" is in the folder named "fno" located at ../fno.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from timeit import default_timer
import os, sys
import json

sys.path.append("../fno")
from utilities3 import *
from Adam import Adam

with open("../data_generation/path.json", 'r', encoding='utf-8') as f:
    JSON = json.load(f)
global_path = JSON["global_path"]
with open("../data_generation/"+global_path, 'r', encoding='utf-8') as f:
    gJSON = json.load(f)
numi = gJSON["numi"]


###
# Copy of SpectralConv1d 
# defined in fourier_1d.py of "fourier_neural_operator"
###
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


###
# Modified version of
# FNO1d defined in fourier_1d.py of "fourier_neural_operator"
###
class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # --------------------------
        # Modification by M. Komatsu
        self.fc1 = nn.Linear(self.width, self.width*2)
        self.fc2 = nn.Linear(self.width*2, 1)
        self.numi = numi
        # --------------------------


    # --------------------------
    # Modification by M. Komatsu
    def forward(self, x):
        ps = x[:,0:self.numi].view(-1,self.numi)
        x = x[:,self.numi:]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    # --------------------------

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

###
# Modified version of FNO1d
###
class FNO1dL(nn.Module):
    def __init__(self, modes, width, convs, ws):
        super(FNO1dL, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        """
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        """
        self.convs = nn.ModuleList(convs)
        self.ws = nn.ModuleList(ws)

        self.fc1 = nn.Linear(self.width, self.width*2)
        self.fc2 = nn.Linear(self.width*2, 1)
        self.numi = numi

    def forward(self, x):
        ps = x[:,0:self.numi].view(-1,self.numi)
        x = x[:,self.numi:]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for conv, ww, in zip(self.convs[:-1], self.ws[:-1]):
            x1 = conv(x)
            x2 = ww(x)
            x = x1 + x2
            x = F.gelu(x)

        x1 = self.convs[-1](x)
        x2 = self.ws[-1](x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


###
# Modified version of
# LowRank1d defined in lowrank_1d.py of "fourier_neural_operator"
###
class LowRank1d(nn.Module):
    def __init__(self, in_channels, out_channels, s, width, rank=1):
        super(LowRank1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s
        self.n = s
        self.rank = rank
        self.phi = DenseNet([2, 64, 128, 256, width*width*rank], torch.nn.GELU)
        self.psi = DenseNet([2, 64, 128, 256, width*width*rank], torch.nn.GELU)
    def forward(self, v, a):
        batch_size = v.shape[0]
        phi_eval = self.phi(a).reshape(batch_size, self.n, self.out_channels, self.in_channels, self.rank)
        psi_eval = self.psi(a).reshape(batch_size, self.n, self.out_channels, self.in_channels, self.rank)
        # --------------------------
        # Modification by M. Komatsu
        temp = torch.einsum('bnoir,bni->bno', psi_eval, v)
        v = torch.einsum('bno,bmoir->bmo', temp, phi_eval) / self.n
        # --------------------------
        return v



###
# Modified version of
# MyNet defined in lowrank_1d.py of "fourier_neural_operator"
###
class LRNO(torch.nn.Module):
    def __init__(self, s, width=32, rank=4):
        super(LRNO, self).__init__()
        self.s = s
        self.width = width
        self.rank = rank
        self.fc0 = nn.Linear(2, self.width)

        self.net1 = LowRank1d(self.width, self.width, s, width, rank=self.rank)
        self.net2 = LowRank1d(self.width, self.width, s, width, rank=self.rank)
        self.net3 = LowRank1d(self.width, self.width, s, width, rank=self.rank)
        self.net4 = LowRank1d(self.width, self.width, s, width, rank=self.rank)
        self.w1 = nn.Linear(self.width, self.width)
        self.w2 = nn.Linear(self.width, self.width)
        self.w3 = nn.Linear(self.width, self.width)
        self.w4 = nn.Linear(self.width, self.width)

        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)
        self.bn4 = torch.nn.BatchNorm1d(self.width)
        # --------------------------
        # Modification by M. Komatsu
        self.fc1 = nn.Linear(self.width, self.width*2)
        self.fc2 = nn.Linear(self.width*2, 1)
        self.numi = numi
        # --------------------------

    # --------------------------
    # Modification by M. Komatsu
    def forward(self, v):
        ps = v[:,0:self.numi].view(-1,self.numi)
        v = v[:,self.numi:]
        grid = self.get_grid(v.shape, v.device)

        v = torch.cat((v, grid), dim=-1)
        v = v.squeeze(-1)
        a = v.clone()
        grid = grid.squeeze(-1)
        batch_size = v.shape[0]
        n = v.shape[1]
        v = self.fc0(v)

        v1 = self.net1(v, a)
        v2 = self.w1(v)
        v = v1+v2
        F.gelu(v)

        v1 = self.net2(v, a)
        v2 = self.w2(v)
        v = v1+v2
        F.gelu(v)

        v1 = self.net3(v, a)
        v2 = self.w3(v)
        v = v1+v2
        F.gelu(v)

        v1 = self.net4(v, a)
        v2 = self.w4(v)
        v = v1+v2
        del a
        v = self.fc1(v)
        F.gelu(v)
        v = self.fc2(v)
        return v
    # --------------------------
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


###
# Modified version of LRNO
###
class LRNOL(torch.nn.Module):
    def __init__(self, nets, ws, bs, width=32, rank=4):
        super(LRNOL, self).__init__()
        self.width = width
        self.rank = rank
        self.fc0 = nn.Linear(2, self.width)
        self.nets = nn.ModuleList(nets)
        self.ws = nn.ModuleList(ws)
        self.bs = nn.ModuleList(bs)
        self.fc1 = nn.Linear(self.width, self.width*2)
        self.fc2 = nn.Linear(self.width*2, 1)
        self.numi = numi


    def forward(self, v):

        ps = v[:,0:self.numi].view(-1,self.numi)
        v = v[:,self.numi:]
        grid = self.get_grid(v.shape, v.device)

        v = torch.cat((v, grid), dim=-1)
        v = v.squeeze(-1)
        a = v.clone()
        grid = grid.squeeze(-1)
        batch_size = v.shape[0]
        n = v.shape[1]
        v = self.fc0(v)

        for net, ww, bb in zip(self.nets[:-1], self.ws[:-1], self.bs):
            v1 = net(v, a)
            v2 = ww(v)
            v = v1 + v2
            v = bb(v.reshape(-1, self.width)).view(batch_size,n,self.width)
            v = F.gelu(v)

        v1 = self.nets[-1](v,a)
        v2 = self.ws[-1](v)
        v = v1+v2
        del a
        v = self.fc1(v)
        F.gelu(v)

        v = self.fc2(v)
        return v

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


