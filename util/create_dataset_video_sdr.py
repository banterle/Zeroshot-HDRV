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
import pandas as pd

import torch
from util.clean_video import *
from util.video_sdr import VideoSDR
from util.io import *
from util.util_io import *
from util.util_np import *

#
#
#
def split_data_from_video_sdr(data_dir, expo_shift = 2.0, group=None, sampling = -2, recs_dir = 'recs', samples_is = 128, scaling = 1.0, format = '.png'):
    global_random_shift = 0.25
    #single image
    ext = os.path.splitext(data_dir)[1]
    
    #video
    v = VideoSDR(data_dir, format, True)
    num_frames = v.getNumFrames()
    lst = []
    
    if (sampling > 0) :
        lst = cleanSequenceRegularSamplingWithJittering(num_frames, sampling)

    elif sampling == -2:
        lst = cleanSequenceWithUniformSampling(v, samples_is, expo_shift, scaling)
        
        if len(lst) == 0:
            sampling = (num_frames // samples_is)
            lst = cleanSequenceRegularSamplingWithJittering(num_frames, sampling, False)

    #create the dataset
    #base_dir = getGlobalPath(data_dir)
    
    print('Frames: ' + str(len(lst)))

    print(data_dir) 
    
    random.seed(42)
    maxOE = 0.0
    frame2 = []
    out2 = []
       
    v.setFrame(0)
    debug = False
    frames_str = []
    frames_next_str = []
    shift_v = []
    filename_oe = ""
    j = 0
    lst2 = []

    for i in range(0, num_frames - 1):
        success, tmp_file_name, tmp_file_name_next, frame = v.getNextFrameWithNext(True)
        
        if scaling > 1.0:
            frame = np.clip(frame * scaling, 0.0, 1.0)
        
        if success:
            maxOE_i = npGetOverexposed(frame, 0.99)
            if maxOE_i > maxOE:
                filename_oe = tmp_file_name
                maxOE = maxOE_i
                j = i
                frame2 = tmp_file_name
            else:
                out2 = tmp_file_name
                
        #success, frame = v.read()
        if (i in lst):
            random_shift = random.random() * global_random_shift
            shift_v.append(random_shift)
                
            frames_str.append(tmp_file_name)#os.path.join(data_dir, tmp_file_name))
            frames_next_str.append(tmp_file_name_next)#os.path.join(data_dir, tmp_file_name_next))
            lst2.append(i)
            if debug:
                npSaveImage(frame, 'images_' + str(i) + '.png')
        
        del frame
        frame = None
    
    
    print('Frames final: ' + str(len(frames_str)))

    fnv = getFilenameOnly(filename_oe)
    filename = os.path.join(recs_dir, fnv + ".png")

    if maxOE > 0:
       print(filename_oe)
       print('Max OE: ' + str(maxOE) + ' ' + str(j) + ' ' + frame2)
       print(filename)
       tmp = Image.open(frame2)
       tmp.save(filename)
       del tmp
       tmp = None
    else:
       tmp = Image.open(out2)
       tmp.save(filename)
       del tmp
       tmp = None

    if group:
        data = []
        n = len(frames_str)
        c = 0
        frameIn = []
        frameIn_next = []
        frameIn_s = []
        group_v = []
        for i in range(0, n):
            for j in range(0, group):
                
                frameIn.append(frames_str[i])
                frameIn_next.append(frames_next_str[i])

                random_shift = random.random() * global_random_shift
                frameIn_s.append(random_shift)
                
                group_v.append(j)

        d = {'Input': frameIn, 'Next': frameIn_next, 'Shift': frameIn_s, 'Group': group_v}
        data = pd.DataFrame(data=d)
        print("Group:" + str(group) + " " + str(len(data)))
        data = [data[i:i + group] for i in range(0, len(data), group)]
        data = pd.concat(data)
    else:
        d = {'Input': frames_str, 'Next': frames_next_str, 'Shift': shift_v}
        data = pd.DataFrame(data=d)

    return data, filename, num_frames

#
#
#
def genDataset(base_dir, args):
    video_folders_tmp = os.listdir(base_dir)
    video_folders_tmp = sorted(video_folders_tmp)

    c = 0
    filename_rec = []
    train_data = []

    for i in range(0, len(video_folders_tmp)):
        v = video_folders_tmp[i]
        if(v.startswith(".")):
            continue
    
        full_path = os.path.join(base_dir, v)
    
        if(os.path.isfile(full_path)):
            continue
        
        print(v)
        train_data_i, filename_rec_i, num_frames = split_data_from_video_sdr(full_path, args.es, group=args.group, sampling = args.sampling, recs_dir = args.recs_dir, samples_is = args.samples_is)
        filename_rec.append(filename_rec_i)
            
        if c == 0:
            train_data = train_data_i
        else:
            tmp = [train_data, train_data_i]
            train_data = pd.concat(tmp)
        
        c += 1

    return train_data, filename_rec
