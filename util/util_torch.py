#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as T
from util.util_io import *

#
#
#
def getImage2Tensor(img):
    img_torch = to_tensor(img)

    if len(img_torch.shape) == 3:
        img_torch = img_torch.unsqueeze(0)

    if torch.cuda.is_available():
        img_torch = img_torch.cuda()
        
    return img_torch

#
#
#
def torchLearningPercentage(frame, expo_shift = 2.0, scaling = 1.0):
    min_val = (0.25 / 255)
    exposure     = np.power(2.0,  expo_shift)
    exposure_inv = np.power(2.0, -expo_shift)
    
    gamma_inv = 1.0 / 2.2
    scale_g = scaling * np.power(exposure, gamma_inv)

    frame_p = torchRound8(torch.clamp(frame * scale_g, 0.0, 1.0))
            
    delta = frame / (frame_p + min_val)
            
    mask = torch.zeros(delta.shape, dtype=torch.float)

    thr = np.power(exposure_inv, gamma_inv)
            
    upper_limit = 1.0 - min_val
    lower_limit = thr + min_val
    mask = torch.where((delta > lower_limit) & (delta < upper_limit), 1.0, mask)
    return torch.mean(mask).item()

#
#
#
def torchSaveImage(x, name):
    img_np = x.data.cpu().numpy().squeeze()
    img = fromTorchToPil(img_np)
    img.save(name)

#
#
#
def crop_center(img,csize):
    y,x = img.shape
    cx = x // 2
    cy = y // 2
    out = img[(cy-csize):(cy+csize),(cx-csize):(cx+csize)]
    return out

#
#
#
def torchDataAugmentation(img, j):
    img_out = []

    if j < 0:
        j = 0

    if j > 7:
        j = 7
    
    if(j == 0):
        img_out = img
    elif (j == 1):
        img_out = T.functional.rotate(img, 90)
    elif (j == 2):
        img_out = T.functional.rotate(img, 180)
    elif (j == 3):
        img_out = T.functional.rotate(img, 270)
    elif (j == 4):
        img_out = T.functional.hflip(img)
    elif (j == 5):
        img_out = T.functional.vflip(img)        
    elif (j == 6):
        img_tmp = T.functional.rotate(img, 90)
        img_out = T.functional.hflip(img_tmp)
        del img_tmp
    elif (j == 7):
        img_tmp = T.functional.rotate(img, 90)
        img_out = T.functional.vflip(img_tmp)
        del img_tmp
        
    return img_out

#
#
#
def torchLum(x):
    y = 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]
    return y
    
#
#
#
def torchRound8(x):
    return torch.round(x * 255.0) / 255.0

#
#
#
def torchChangeExposure(x, f, gamma = 2.2):
    exposure = np.power(2.0, f)
    exposure_invGamma = np.power(exposure, 1.0 / gamma)
    out = torch.clamp(x * exposure_invGamma, 0.0, 1.0)
    
    return out

#
#
#
def torchSoftMask(x,  thr = 1.0, s = 0.15):
    d = x - thr
    s2_sq = s * s * 2.0
    out = torch.exp(-(d * d) / s2_sq)
    return out
    
#
#
#
def getGoodPixels(x):
    return torch.clamp(torchSoftMask(o0, 0.5, 0.125)*10.0, 0.0, 1.0)
 
#
#
#
def torchReadImg(fname, index = 0, group = None):
    img = Image.open(fname)
    index_t =  index % group
    if( group != None ):
        img = dataAugmentation(img, index_t)
    x = to_tensor(img)

    return x
