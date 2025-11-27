#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import numpy as np
from PIL import Image
import cv2

#
#
#
def createVideo(filename, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
    return out

#
#
#
def readPIL(str):
    return Image.open(str)

#
#
#
def switchBGR(frame):
    s = frame.shape
    out = np.zeros((s[0], s[1], s[2]), dtype = np.float32)
    out[:,:,0] = frame[:,:,2]
    out[:,:,1] = frame[:,:,1]
    out[:,:,2] = frame[:,:,0]
    return out

#
#
#
def fromNPtoVideoFrame(frame, BGR = False, CV2FMT = False):

    frame = np.round(frame * 255.0)
    frame = frame.astype(dtype = np.uint8)
    
    s = frame.shape
    if CV2FMT:
        out = np.zeros((s[0], s[1], s[2]), dtype = np.uint8)
        if BGR:
            out[:,:,0] = frame[:,:,2]
            out[:,:,1] = frame[:,:,1]
            out[:,:,2] = frame[:,:,0]
        else:
            out = frame
    else:
        out = np.zeros((s[1], s[2], s[0]), dtype = np.uint8)
    
        if BGR:
            out[:,:,0] = frame[2,:,:]
            out[:,:,1] = frame[1,:,:]
            out[:,:,2] = frame[0,:,:]
        else:
            out[:,:,0] = frame[0,:,:]
            out[:,:,1] = frame[1,:,:]
            out[:,:,2] = frame[2,:,:]
    return out

#
#
#
def fromVideoFrameToNP(frame, BGR = False):
    
    frame = frame.astype(dtype = np.float32)
    frame = frame / 255.0
    if BGR:
        out = np.zeros(frame.shape, dtype = np.float32)
        out[:,:,0] = frame[:,:,2]
        out[:,:,1] = frame[:,:,1]
        out[:,:,2] = frame[:,:,0]
    else:
        out = frame
    return out

#
#
#
def fromNPtoPIL(img):

    img_clipped = np.clip(img, 0.0, 1.0)
    formatted = (img_clipped * 255.0).astype('uint8')
    s = formatted.shape
    
    if (s[0] < s[1]) and (s[0] < s[2]):
        out = np.zeros((s[1], s[2], s[0]), dtype = np.uint8)
        out[:,:,0] = formatted[0,:,:]
        out[:,:,1] = formatted[1,:,:]
        out[:,:,2] = formatted[2,:,:]
        img_pil = Image.fromarray(out)
    else:
        img_pil = Image.fromarray(formatted)
        
    return img_pil

#
#
#
def fromPILtoNP(img, bNorm = False):
    img_np = np.array(img);
    img_np = img_np.astype('float32')
    if bNorm:
       img_np /= 255.0
    return img_np

#
#
#
def fromTorchToNP(p):
    sz = p.shape
    out = np.zeros((sz[1], sz[2], sz[0]))
    for i in range(0, sz[0]):
        tmp = p[i, 0:sz[1], 0:sz[2]]
        out[:,:,i] = tmp
    return out

#
#
#
def fromTorchToPil(p):
    sz = p.shape
    if len(sz) == 2:
        out = np.zeros((sz[0], sz[1], 3))
        for i in range(0, 3):
            out[:,:,i] = p
    else:
        out = np.zeros((sz[1], sz[2], sz[0]))
        for i in range(0, sz[0]):
            tmp = p[i, 0:sz[1], 0:sz[2]]
            out[:,:,i] = tmp
    return fromNPtoPIL(out)

#
#
#
def npReadHDRWithCV(filename, bBGR = True):
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
   
    out = np.zeros(img.shape, dtype = np.float32)

    if bBGR:
        out[:,:,0] = img[:,:,2]
        out[:,:,1] = img[:,:,1]
        out[:,:,2] = img[:,:,0]
    else:
        out[:,:,0] = img[:,:,0]
        out[:,:,1] = img[:,:,1]
        out[:,:,2] = img[:,:,2]
        
    return out

#
#
#
def npImgRead(path):
    ext = (os.path.splitext(path)[1]).lower()
    if (ext == '.exr') or (ext == '.hdr') or (ext == '.pfm'):
        return npReadHDRWithCV(path)
    else:
        tmp = readPIL(path).convert('RGB')
        img = fromPILtoNP(tmp) / 255.0
        
        if img.shape[2] == 4:
            img = img[:,:,0:3]
        
    return img
