#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import sys
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import numpy as np
from util.clean_video import *
from util.video_sdr import *
from util.util_io import *
from util.util_np import *
from util.util_torch import *

from PIL import Image   

#
#
#
class SDRDataset(Dataset):

    #
    #
    #
    def __init__(self, data, group = None,  expo_shift = 2.0, bRandom = False, scale = 1.0, area = -1, temporal = True):
        self.data = data
        self.expo_shift = expo_shift
        self.group = group
        self.bRandom = bRandom
        self.epoch = 0
        self.scale = scale
        self.patchSize = 128
        self.bTemporal = temporal

        self.cache = CachedImages()

        if area == -1:
            self.numPatches = 4
        else:
            self.numPatches = area // (self.patchSize * self.patchSize)
            if self.numPatches <= 0:
                self.numPatches = 1

        print('Number of patches: ' + str(self.numPatches))
            
        if bRandom:
            random.seed(42)
        
    #
    #
    #
    def read_img(self, fname, index = 0, group = None):
        img = Image.open(fname)
    
        if( group != None ):
            img = torchDataAugmentation(img, index)
        
        x = to_tensor(img)
        x_shape = x.shape
    
        #remove alpha channel
        if (x_shape[0] == 4):
            x = x[0:3,:,:]

        return x            

    #
    #
    #
    def __getitem__(self, index):

        #
        #
        #
        sample = self.data.iloc[index // self.numPatches]
            

        index_t = 0
        if self.group != None:
            if self.group > 1:
                index_t = sample.Group

        if self.bRandom:
                shift = random.random() * 0.25
        else:
            shift = sample.Shift
            
        img_sdr = self.read_img(sample.Input, index_t, self.group) * self.scale

        if self.bTemporal:
            img_sdr_n = self.read_img(sample.Next, index_t, self.group) * self.scale
                    
        if (self.patchSize < img_sdr.shape[1]) and (self.patchSize < img_sdr.shape[2]):
            limit_y = img_sdr.shape[1] - self.patchSize - 1
            limit_x = img_sdr.shape[2] - self.patchSize - 1
            bestAvg = 0.0
            xb = -1
            yb = -1
            bFlag = True
            count = 0

            while bFlag:
                x = int(np.round(np.random.rand() * limit_x))
                y = int(np.round(np.random.rand() * limit_y))
            
                tmp = img_sdr[:,y:(y+self.patchSize),x:(x+self.patchSize)]

                avg = torchLearningPercentage(tmp, self.expo_shift).item()
                bFlag = (avg < 0.3) or np.isnan(avg)

                if bFlag:
                    if avg > bestAvg:
                        xb = x
                        yb = y
                        bestAvg = avg

                if count >= 100:
                    #print('Fail: ' + str([xb,yb, bestAvg]))
                    x = xb
                    y = yb
                    avg = bestAvg
                    bFlag = False            
                count += 1

            #torchSaveImage(img_sdr[:,y:(y+self.patchSize), x:(x+self.patchSize)], 'epoch_' + str(self.epoch) + '_patch_'+str(index)+'_lp_'+str(avg)+'_'+str(count)+'.png')

            img_sdr = img_sdr[:,y:(y+self.patchSize), x:(x+self.patchSize)]
            if self.bTemporal:
                img_sdr_n = img_sdr_n[:,y:(y+self.patchSize),x:(x+self.patchSize)]
        
        o0 = torchRound8(torchChangeExposure(img_sdr, shift, 2.2))
        f0 = torchRound8(torchChangeExposure(img_sdr, shift + self.expo_shift, 2.2))

        if self.bTemporal:
            o0_n = torchRound8(torchChangeExposure(img_sdr_n, shift, 2.2))
            f0_n = torchRound8(torchChangeExposure(img_sdr_n, shift + self.expo_shift, 2.2))
        else:
            o0_n = []
            f0_n = []
        
        return f0, o0, o0_n, f0_n

    #
    #
    #
    def __len__(self):
        return len(self.data) * self.numPatches
