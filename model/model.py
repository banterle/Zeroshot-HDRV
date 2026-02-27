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

#
#
#
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.f = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU()
                        )
                               
    def forward(self, x):
        return self.f(x)

#
#
#
class UDown(nn.Module):

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

    def forward(self, x):
        return self.down(x)

#
#
#
class UUp(nn.Module):

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
    def forward(self, x):
        return self.up(x)

def merge(x, y):
    return torch.cat((x, y), 1)
    
              
#
#Network
#
class UNet(nn.Module):
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

        self.d0 = UDown(n_input, r[0], True) #512x512
        self.d1 = UDown(r[0], r[1])          #256x256
        self.d2 = UDown(r[1], r[2])          #128x128
        self.d3 = UDown(r[2], r[3])          #64x64
        self.bFull = bFull
       
        if bFull:
            self.d4 = UDown(r[3], r[4])
            self.u4 = UUp(r[4], r[3])
            self.u3 = UUp(2 * r[3], r[2])
        else:
            self.u3 = UUp(1 * r[3], r[2])
            
        self.u2 = UUp(2 * r[2], r[1])
        self.u1 = UUp(2 * r[1], r[0])
        self.u0 = UUp(2 * r[0], n_output, True)
        
    #
    #
    #
    def divider(self):
        return 16        

    #
    #
    #
    def forward(self, v_input, mask = None):    

        o0 = self.d0(v_input)
        o1 = self.d1(o0)
        o2 = self.d2(o1)
        o3 = self.d3(o2)

        #32x32
        if self.bFull:
            o4 = self.d4(o3)
            u4 = self.u4(o4)
            o3 = Crop2D(o3, u4)
            u3 = self.u3(merge(u4, o3))
        else:
            u3 = self.u3(o3)
                    
        o2 = Crop2D(o2, u3)
        u2 = self.u2(merge(u3, o2))
        
        o1 = Crop2D(o1, u2)
        u1 = self.u1(merge(u2, o1))
        
        o0 = Crop2D(o0, u1)
        u0 = self.u0(merge(u1, o0))
                                  
        return u0
