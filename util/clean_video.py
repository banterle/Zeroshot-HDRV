#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from util.util_io import *
from util.video_sdr import *
from util.distribution1D import *
from util.util_np import npGetAverageOverexposed, npGetGoodPixels, npRound8

#
#   regular sampling
#
def cleanSequenceRegularSampling(n, fq = 6):
    frames_vec = []
    for i in range(0,n):
        if (i % fq) == 0:
            frames_vec.append(i)
            
    return frames_vec

#
#   regular sampling + jittering
#
def cleanSequenceRegularSamplingWithJittering(n, fq = 6):
    random.seed(42)
    frames_vec = []
    for i in range(0,n):
        if (i % fq) == 0:
            index = random.randint(0, fq - 1)
            j = i + index
            if(j < n):
                frames_vec.append(j)
            else:
                frames_vec.append(i)

    return frames_vec
           
#
#  cleanSequenceWithUniformSampling
#
def cleanSequenceWithUniformSampling(video_obj, nSamples, expo_shift = 2.0, scaling = 1.0, bDebug = False):
    
    frames_oe = []
    n = video_obj.n
    
    thr = 0.05
    for i in range(0, n):
        success, tmp_file_name, frame = video_obj.getNextFrame(True)
        oe_avg = npGetGoodPixels(frame, thr)
        frames_oe.append(oe_avg)
        
    frames_vec = []

    if bDebug:
        sns.distplot(frames_oe, kde=True, rug=True, bins=32)
        plt.savefig('hist_oe.png')

    if nSamples > n:
        nSamples = n

    print('nSamples: ' + str(nSamples))

    if nSamples > 30:
        n_6 = n // 6
        if nSamples > n_6:
            nSamples = n_6
        
    ism = Distribution1D(frames_oe, True)
    frames_vec = ism.sampleRangeWithLimit(nSamples, 0.0, 1.0, 0.0)
    frames_vec.sort()
    
    print(frames_vec)

    bRemoveClones = False
                
    if bRemoveClones:
        frames_vec.sort()
        frames_out = []
        thr = 6
        for v in frames_vec:
            if v not in frames_out:
                if len(frames_out) > 0:
                    last = frames_out[-1]
                    if(abs(v-last) >= thr):
                        frames_out.append(v)
                else:
                    frames_out.append(v)
    else:
        frames_out = frames_vec
    
    print('Total Frames: ' + str(n))
    print('Picked Frames: ' + str(len(frames_out)))
    
    return frames_out
