#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import numpy as np
from util.util_io import fromNPtoPIL

#
#
#
def addBorder(img, div = 16):
    sz_sdr = img.shape
    bFlag = False
    bX = False
    bY = False
            
    if (sz_sdr[0] % div) > 0:
        bFlag = True
        bX = True
                
    if (sz_sdr[1] % div) > 0:
        bFlag = True
        bY = True

    if bFlag:
        sz0 = sz_sdr[0]
        sz1 = sz_sdr[1]
                
        if bX:
            sz0 = ((sz_sdr[0] // div) + 1) * div
            
        if bY:
            sz1 = ((sz_sdr[1] // div) + 1) * div
                    
        img_new = np.zeros((sz0, sz1, sz_sdr[2]), dtype = img.dtype)
        img_new[0:sz_sdr[0], 0:sz_sdr[1], :] = img
        return img_new, True
    else:
        return img, False

#
#
#
def npRound8(x):
    return np.round(255.0 * x) / 255.0

#
#
#
def npSaveImage(x, name):
    img = fromNPtoPIL(x)
    img.save(name)
   
#
#
#
def npApplyGamma(frame, fExp = 1.0, fGamma = 2.2, bFstop = False):
    if bFstop:
        fExp = np.power(2.0, fExp)
    
    ret = frame * fExp
    np.power(ret, 1.0 / fGamma, out = ret)
    np.clip(ret, 0.0, 1.0, out = ret)
    return ret

#
#
#
def npMSE(img1, img2):
    mse = np.mean(np.power((img1 - img2), 2.0))
    return mse

#
#
#
def npMAE(img1, img2):
    mae = np.mean(np.abs(img1 - img2))
    return mae

#
#
#
def npPSNR(img1, img2):
    mse = MSE(img1, img2)
    return 10.0 * np.log10(1.0 / mse)

#
#
#
def npLuminance(x, mode = 'CIE_Y'):
    r,c,col = x.shape
    
    if col == 3:
        if mode == 'CIE_Y':
            y = 0.2126 * x[:,:,0] + 0.7152 * x[:,:,1] + 0.0722 * x[:,:,2]
        elif mode == 'mean':
            y = (x[:,:,0] + x[:,:,1] + x[:,:,2]) / 3.0
    else:
        y = []
        
    return y
    
#
#
#
def npChangeExposure(x, f = 0.0, gamma = 2.2):
    exposure = np.power(2.0, f)
    invGamma = 1.0 / gamma
    exposure_invGamma = np.power(exposure, invGamma)
    y = x * exposure_invGamma
    y = np.clip(y, 0.0, 1.0)
    return y
    
#
#
#
def npSetExposureGamma(x, f = 0.0, gamma = 2.2):
    exposure = np.power(2.0, f)
    invGamma = 1.0 / gamma
    out = np.clip(np.power(x * exposure, invGamma), 0.0, 1.0)
    return out
    
#
#
#
def npGetOverexposed(img, thr = 0.95):
    #lum = npLuminance(img)
    tmp = img.flatten()
    index = np.where(tmp > thr)
    index = index[0]
    return len(index) / len(tmp)
    
#
#
#
def npGetOverexposed2(x, thr = 0.95):
    sz = x.shape
    mask = np.zeros(sz)
    mask[np.where(x > thr)] = 1.0
    t0 = np.maximum(mask[:,:,0], mask[:,:,1])
    t1 = np.maximum(t0, mask[:,:,2])
    return np.mean(mask)
    
#
#
#
def npComputeMask(x, thr = 0.95, bType = False, bSingle = False, bRel = True):
    sz = x.shape
    mask = np.zeros(sz)
    if bRel:
        mask[np.where(x > thr)] = 1.0
    else:
        mask[np.where(x < thr)] = 1.0

    t0 = np.maximum(mask[:,:,0], mask[:,:,1])
    t1 = np.maximum(t0, mask[:,:,2])
    
    if bType:
       t1 = 1.0 - t1
       
    if bSingle:
        mask = t1
    else:
        for i in range(0,3):
            mask[:,:,i] = t1
        
    return mask
    
#
#
#
def npComputeMaskOE(x, thr = 0.95, bType = False):
    sz = x.shape
    mask = np.zeros(sz)
    thr_inv = 1.0 - thr
    mask[np.where(x > thr)] = 1.0
    mask[np.where(x < thr_inv)] = 1.0
    
    t0 = np.maximum(mask[:,:,0], mask[:,:,1])
    t1 = np.maximum(t0, mask[:,:,2])
    
    if bType:
       t1 = 1.0 - t1
       
    for i in range(0,3):
        mask[:,:,i] = t1
        
    return mask
 
#
#
#
def npComputeSoftMask(x, thr = 0.95):
    sz = x.shape
    mask = np.zeros(sz)
    t0 = (x[:,:,0] + x[:,:,1] + x[:,:,2]) / 3.0
    for i in range(0,3):
        mask[:,:,i] = t0
    
    return 1.0 - mask
    
#
#
#
def npGetAverageOverexposed(x):
    mask = npComputeMask(x, 0.9, False, True)
    return np.mean(mask)

#
#
#
def npGetGoodPixels(x, thr = 0.05):
    sz = x.shape
    mask = np.zeros(sz)
    thr_i = 1.0 - thr
    mask[np.where((x > thr) & (x < thr_i))] = 1.0
    t0 = np.minimum(mask[:,:,0], mask[:,:,1])
    t1 = np.minimum(t0, mask[:,:,2])
    return np.mean(mask)

#
#
#
def npNormalize(img):
    out = (img - np.min(img)) / (np.max(img) - np.min(img))
    return out

#
#
#
def npFromFloatToUint8(img):
    img *= 255
    formatted = np.clip(img, a_min = 0, a_max = 255)
    formatted = formatted.astype('uint8')
    return formatted

#
#
#
def fromNPtoVideoFrame(frame, fGamma = 2.2, BGR = False):
    if fGamma > 0.0:
        np.power(frame, 1.0 / fGamma, out = frame)
    
    frame = np.clip(np.round(frame * 255.0), 0.0, 255.0)
    frame = frame.astype(dtype = np.uint8)
    
    s = frame.shape
    out = np.zeros(s, dtype = np.uint8)
    
    if BGR:
        out[:,:,0] = frame[:,:,2]
        out[:,:,1] = frame[:,:,1]
        out[:,:,2] = frame[:,:,0]
    else:
        out[:,:,0] = frame[:,:,0]
        out[:,:,1] = frame[:,:,1]
        out[:,:,2] = frame[:,:,2]
        
    return out
