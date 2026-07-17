 #
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import torch

import numpy as np
from torch import nn

from model.uber_pooling_layer import *
from model.cropping_layer import *
from model.model import *

#
#
#
class UDown_t(nn.Module):

    def __init__(self, in_channels, out_channels, bStart = False):
        super().__init__()
        if bStart:
           self.down = nn.Sequential(
                Block(in_channels, out_channels)
           )
        else:
           self.down = nn.Sequential(
               UberPool(),
               Block(in_channels, out_channels)
           )
           
        self.td = nn.Parameter(torch.randn(1, out_channels, 1, 1).float())

    #
    #
    #
    def forward(self, x, t):
        return self.down(x) + self.td * float(t)

#
#
#
class UUp_t(nn.Module):

    def __init__(self, in_channels, out_channels, bEnd = False):
        super().__init__()
        if bEnd:
           half = in_channels // 2
           self.up = nn.Sequential(
                                Block(in_channels, half),
                                nn.Conv2d(half, out_channels, kernel_size=1, padding=0),
                                nn.Sigmoid()
                                )
        else:
           self.up = nn.Sequential(
                                   Block(in_channels, out_channels),
                                   nn.Upsample(scale_factor = 2.0, mode = 'bilinear', align_corners = True)
                                )
                                
        self.tu = nn.Parameter(torch.randn(1, in_channels // 2, 1, 1).float())
        self.tu2 = nn.Parameter(torch.randn(1, in_channels, 1, 1).float())


    #
    #
    #
    def forward(self, x, y = None, t = 2.0):
    
        if y == None:
            return self.up(x + self.tu2 * float(t))
        else:
            o2 = Crop2D(y, x)
            xtut = x + self.tu * float(t)
            xtuty = merge(xtut, y)
            return self.up(xtuty)

#
#Network
#
class UNet_t(nn.Module):
    
    #
    #
    #
    def __init__(self, n_input=3, n_output=3, fstop = 2.0, bFull = False, maskVal = 0.5):
        super().__init__()

        self.maskVal = maskVal
        self.gamma = 2.2
        self.inv_gamma = 1.0 / self.gamma
        
        self.es = np.power(2.0, fstop)
        self.es_g = np.power(self.es, self.inv_gamma)

        if fstop > 0.0:
            self.w = 0.0
            self.sigma = 0.0001
        else:
            self.w = 1.0
            self.sigma = 0.15

        r = [64, 128, 256, 512, 512]

        self.d0 = UDown_t(n_input, r[0], True) #512x512
        self.d1 = UDown_t(r[0], r[1])          #256x256
        self.d2 = UDown_t(r[1], r[2])          #128x128
        self.d3 = UDown_t(r[2], r[3])          #64x64
        self.bFull = bFull
       
        if bFull:
            self.d4 = UDown_t(r[3], r[4])
            self.u4 = UUp_t(r[4], r[3])
            self.u3 = UUp_t(2 * r[3], r[2])
        else:
            self.u3 = UUp_t(r[3], r[2])
            
        self.u2 = UUp_t(2 * r[2], r[1])
        self.u1 = UUp_t(2 * r[1], r[0])
        self.u0 = UUp_t(2 * r[0], n_output, True)
        
    #
    #
    #
    def divider(self):
        return 16        

    #
    #
    #
    def forward(self, v_input, mask = None, t = 2.0):

        o0 = self.d0(v_input, t)
        o1 = self.d1(o0, t)
        o2 = self.d2(o1, t)
        o3 = self.d3(o2, t)

        #32x32
        if self.bFull:
            o4 = self.d4(o3, t)
            u4 = self.u4(o4, None, t)
            u3 = self.u3(u4, o3, t)
        else:
            u3 = self.u3(o3, None, t)
                    
        u2 = self.u2(u3, o2, t)
        u1 = self.u1(u2, o1, t)
        u0 = self.u0(u1, o0, t)
                                  
        return u0


if __name__ == "__main__":
    model = UNet_t(3,3)
    
    img = torch.zeros((1,3,256,256), dtype = torch.float32)
    print(out)
    print(out.shape)
