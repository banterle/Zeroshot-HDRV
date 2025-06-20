#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import torch
import copy
from torch import nn
import numpy as np
import sys
import glob2
import os
import re

#from model.model_attention import UNet
from model.model import UNet
from util.util_io import fromTorchToNP

#
#
#
class UNetUD(nn.Module):
    #
    #
    #
    def __init__(self, n_input=3, n_output=3, fstop = 2.0, ckpt_str = None, mode = 4):
        super().__init__()
        
        self.bLoad = False

        self.min_val = 0.25 / 255

        self.bUNet = (mode == 0)
        
        self.bCuda = torch.cuda.is_available()      
 
        self.exposure = np.power(2.0, fstop)
        self.exposure_inv = np.power(2.0, -fstop)
        self.exposure_gamma = np.power(self.exposure, 1.0 / 2.2)
        self.exposure_inv_gamma = np.power(self.exposure_inv, 1.0 / 2.2)
        self.mode = mode
        self.temp = 4
        self.sampling = -2
        
        if ckpt_str != None:
            self.load_weights(ckpt_str)
        else:
            if self.bUNet:
                self.U = UNet(n_input, n_output, fstop =  fstop)
            self.D = UNet(n_input, n_output, fstop = -fstop)
            self.epoch = 0
            
        self.eval()
        self.fstop = fstop
        
    #
    #
    #
    def load_weights(self, ckpt_str):
        ckpt_str_ext = (os.path.splitext(ckpt_str)[1]).lower()
        
        self.bLoad = True

        #load
        if ckpt_str_ext == '.pth':
            if self.bCuda:
                ckpt = torch.load(ckpt_str)
            else:
                ckpt = torch.load(ckpt_str, map_location=torch.device('cpu'))
        else:
            ckpt_dir = os.path.join(ckpt_str, 'ckpt')
            ckpts = glob2.glob(os.path.join(ckpt_dir, '*.pth'))

            def get_epoch(ckpt_url):
                s = re.findall("ckpt_e(\\d+).pth", ckpt_url)
                epoch = int(s[0]) if s else -1
                return epoch, ckpt_url

            try:
                start_epoch, ckpt = max(get_epoch(c) for c in ckpts)
                print('Checkpoint:', ckpt)
        
                if self.bCuda:
                    ckpt = torch.load(ckpt, weights_only=True)
                else:
                    ckpt = torch.load(ckpt, weights_only=True, map_location=torch.device('cpu'))
            except:
                print('Failed loading ckpt weights.')
                return 0

        #allocate
        n_input = ckpt['n_input']
        n_output = ckpt['n_output']
        
        self.temp = ckpt['temp']
        self.sampling = ckpt['sampling']

        self.mode = ckpt['mode']
        self.fstop = ckpt['es']

        if self.mode == 0:
            self.U = UNet(n_input, n_output, fstop =  self.fstop)
        self.D = UNet(n_input, n_output, fstop = -self.fstop)
            
        self.load_state_dict(ckpt['model'])
        self.epoch = ckpt['epoch']
        self.best_mse = ckpt['mse']
        print('Loaded Epoch: ' + str(self.epoch))
        
        self.bUNet = (self.mode == 0)

    #
    #
    #
    def getString(self):
        if self.bLoad:
            return '_m' + str(self.mode) + '_t' + str(self.temp) + '_s' + str(self.sampling)
        else:
            return '_m' + str(self.mode)

    #
    #
    #
    def forward(self, input, mask = None):
        rec_u = self.fD(input, mask)
        rec_d = self.fU(input, mask)
        return rec_u, rec_d

    #
    #
    #
    def fU(self, input, mask = None):
        if self.bUNet:
            out = self.U(input, mask)
        else:
            out = input * self.exposure_gamma
            out = torch.clamp(out, 0.0, 1.0)
        return out
            
    #
    #
    #
    def fD(self, input, mask = None):
        out = self.D(input, mask)
        return out

    #
    #
    #
    def getExpD(self, input):
        
        if (self.mode == 0) or (self.mode == 1):
            return self.fD(input)
        
        elif (self.mode == 2) or (self.mode == 4):
            delta_mul = self.fD(input)
            return input * delta_mul
            
    #
    #
    #
    def getExpU(self, input):
        if (self.mode == 0) or (self.mode == 1) or (self.mode == 2):
            return self.fU(input)
        
        elif (self.mode == 4):
            delta_mul = self.fD(input)
            return torch.clamp(input / (delta_mul + self.min_val), 0.0, 1.0)
   #
    #
    #
    def addPadding(self, img):
        sz = img.shape

        div = self.D.divider()
        sz2 = int(np.ceil(sz[2] / div) * div)
        sz3 = int(np.ceil(sz[3] / div) * div)

        if (sz2 == sz[2]) and (sz3 == sz[3]):
            return img

        out = torch.zeros((sz[0], sz[1], sz2, sz3), device = img.device,  dtype = img.dtype)
        out[:,:,0:sz[2],0:sz[3]] = img
        return out

    #
    #
    #
    def predict(self, img, n_exp_down = 2, n_exp_up = 2):
        self.eval()

        if(len(img.shape) == 3):
            img = img.unsqueeze(0)
        
        sz_ori = img.shape

        exposures = []
        exposures_times = []
        
        exposures.append(fromTorchToNP(img.data.cpu().numpy().squeeze()))
        exposures_times.append(1.0)
        
        img_i = img
        for i in range(1, (n_exp_down + 1)):
            img_i = self.addPadding(img_i)
            img_i = self.getExpD(img_i)
            img_i = img_i[:,:,0:sz_ori[2],0:sz_ori[3]]
            exposures.append(fromTorchToNP(img_i.data.cpu().numpy().squeeze()))
            exposure_time = np.power(2.0, float(-i * self.fstop))
            exposures_times.append(exposure_time)

        img_i = img
        for i in range(1, (n_exp_up + 1)):
            img_i = self.addPadding(img_i)
            img_i = self.getExpU(img_i)
            img_i = img_i[:,:,0:sz_ori[2],0:sz_ori[3]]
            exposures.append(fromTorchToNP(img_i.data.cpu().numpy().squeeze()))
            exposure_time = np.power(2.0, float( i * self.fstop))
            exposures_times.append(exposure_time)
            
        return exposures, exposures_times

    #
    #
    #
    def predict4(self, img, bTiming = False):
        self.eval()
                
        if(len(img.shape) == 3):
            img = img.unsqueeze(0)

        sz_ori = img.shape
        img = self.addPadding(img)

        t_start = 0;
        t_end = 0;
        
        if bTiming:
            t_start = time.time();

        with torch.no_grad():
        
            if (self.mode == 0) or (self.mode == 1):
                img_d = self.fD(img)
                img_dd = self.fD(img_d)
               
                img_u = self.fU(img)
                img_uu = self.fU(img_u)
                                
            elif (self.mode == 2):
                img_d = img * self.fD(img)
                img_dd = img_d * self.fD(img_d)
                
                img_u = self.fU(img)
                img_uu = self.fU(img_u)
                                
            elif (self.mode == 4):
                d1 = self.fD(img)

                #d1[d1<self.exposure_inv_gamma] = self.exposure_inv_gamma
                img_d = img * d1
                                
                d2 = self.fD(img_d)
                #d2[d2<self.exposure_inv_gamma] = self.exposure_inv_gamma
                img_dd = img_d * d2
            
                img_u = torch.clamp(img / (d1 + self.min_val), 0.0, 1.0)
                
                d3 = self.fD(img_u)
                #d3[d3<self.exposure_inv_gamma] = self.exposure_inv_gamma
                img_uu = torch.clamp(img_u / (d3 + self.min_val), 0.0, 1.0)
              
            if bTiming:
                t_end = time.time();
            
            #download from the GPU; perhaps this could be avoided or delayed
            img_u = img_u[:,:,0:sz_ori[2],0:sz_ori[3]]
            img_uu = img_uu[:,:,0:sz_ori[2],0:sz_ori[3]]
            img_d = img_d[:,:,0:sz_ori[2],0:sz_ori[3]]
            img_dd = img_dd[:,:,0:sz_ori[2],0:sz_ori[3]]

            d1 = d1[:,:,0:sz_ori[2],0:sz_ori[3]]
            d2 = d2[:,:,0:sz_ori[2],0:sz_ori[3]]
            d3 = d3[:,:,0:sz_ori[2],0:sz_ori[3]]

            img_u = img_u.data.cpu().numpy().squeeze()
            img_uu = img_uu.data.cpu().numpy().squeeze()
            img_d = img_d.data.cpu().numpy().squeeze()
            img_dd = img_dd.data.cpu().numpy().squeeze()
            d1 = d1.data.cpu().numpy().squeeze()
            d2 = d2.data.cpu().numpy().squeeze()
            d3 = d3.data.cpu().numpy().squeeze()

        if bTiming:
            t_total = t_end - t_start
            print('Timing: ' + str(t_total))
            
        return img_dd, img_d, img_u, img_uu, d1, d2, d3
