#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import copy
import torch

import numpy as np
from torch import nn

#
#
#
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, std = 1, bFunc = 0):
        super().__init__()
                
        if bFunc == 0:
            self.f = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride = std),
                            nn.ReLU()
                            )
        elif bFunc == 1:
            self.f = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride = std),
                            nn.LeakyReLU(0.2, True)
                            )
        elif bFunc == 2:
            self.f = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride = std),
                            nn.Sigmoid()
                            )
            
                               
    def forward(self, x):
        return self.f(x)


class UUp(nn.Module):

    def __init__(self, in_channels, out_channels, bEnd = False):
        super().__init__()
        if bEnd:
           self.up = nn.Sequential(
                                Block(in_channels, in_channels),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
                                #nn.Sigmoid()                                
                                )
        else:
           self.up = nn.Sequential(
                                   Block(in_channels, out_channels),
                                   nn.Upsample(scale_factor = 2.0, mode = 'bilinear', align_corners = True)
                                   #Block(out_channels, out_channels)
                                )
    def forward(self, x):
        return self.up(x)

#
#
#
def merge(x, y):
    return torch.cat((x, y), 1)  
              
#
#Network
#
class UNet(nn.Module):
    def __init__(self, n_input=3, n_output=3, bFull = False, fstop = 2.0, maskVal = 0.5):
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

        self.up = nn.Upsample(scale_factor = 2.0, mode = 'bilinear', align_corners = True)
        self.max2d = nn.MaxPool2d((2,2), stride =(2,2))

        self.d0 = Block(n_input*2, 64, bFunc = 1)    #512x512
        self.d1 = Block(64, 128, bFunc = 1)     #256x256
        self.d2 = Block(128, 256, bFunc = 1)    #128x128
        self.d3 = Block(256, 512, bFunc = 1)    #64x64
        self.d4 = Block(512, 512, bFunc = 1)    #32x32
        
        self.d5 = Block(512, 512, 2, bFunc = 1)    #16x16
        
        self.u4 = Block(1024, 512)      #32x32
        self.u3 = Block(1024, 256)       #64x64
        self.u2 = Block(512, 128)       #128x128
        self.u1 = Block(256, 64)        #256x256
        self.u0 = Block(128, 3, bFunc = 2)          #512x512

    #
    #
    #
    def divider(self):
        return 64

    #
    #
    #
    def forward_t(self, v_input):
        o0 = self.d0(v_input)
        o1 = self.d1(o0)
        o1=self.max2d(o1)
        o2 = self.d2(o1)
        o2=self.max2d(o2)
        o3 = self.d3(o2)
        o3=self.max2d(o3)
        o4 = self.d4(o3)
        o4=self.max2d(o4)

        m0 = self.d5(o4)
        
        m0u = self.up(m0)
        u0 = self.u4(merge(m0u, o4))

        u0 = self.up(u0)
        u1 = self.u3(merge(u0, o3))

        u1 = self.up(u1)
        u2 = self.u2(merge(u1, o2)) 

        u2 = self.up(u2)
        u3 = self.u1(merge(u2, o1))

        u3 = self.up(u3)
        u4 = self.u0(merge(u3, o0))

        if not self.training:
            u4 = u4.clamp(0.0, 1.0)
                                  
        return u4
    #
    #
    #
    def forward(self, v_input, mask = None):    
        v_input2 = merge(v_input, torch.rand(v_input.shape, device = v_input.device, dtype = v_input.dtype))
        tmp = self.forward_t(v_input2)

        tmp = self.forward_t(merge(v_input, tmp))
        tmp = self.forward_t(merge(v_input, tmp))
        tmp = self.forward_t(merge(v_input, tmp))

        return tmp


if __name__ == '__main__':
    model = UNet(3,3)
    out = model(torch.zeros(1,3,512,512))
    print(out.shape)
