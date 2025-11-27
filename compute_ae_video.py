import os
import sys

import numpy as np
import cv2

from util.util_io import npReadHDRWithCV
from util.util_np import npLuminance, npApplyGamma, fromNPtoVideoFrame
from util.io import mkdir_s

#
#
#
def computeAutomaticExposure(img, pML, alpha = 0.5, bCentral = 'Central'):
    width, height, cols  = img.shape

    L = npLuminance(img)

    if bCentral == 'Mid9':
        sw = width // 3
        sh = height // 3
        ew = (width * 2) // 3
        eh = (height * 2) // 30
        L = L[sw:ew,sh:eh]

    if bCentral == 'Central':
        value = min([width, height])

        wh = width // 2
        vh = value // 2
        sw = wh - vh
        ew = wh + vh
        L = L[sw:ew,:]

    
    L1 = L.flatten()
    L1 = np.sort(L1)
    n = len(L1)
    n_half = int(np.round(n * 0.5))

    mL0 = np.mean(L) * 4
    mL1 = L1[(n*90)//100]
    mL = np.max([mL0, mL1])
    #print([mL0, mL1, mL])

    if pML > 0.0:
       mL = mL * alpha + pML * (1.0 - alpha)
    fExp = 1.0 / (mL);

    return mL, fExp
       
#
#
#
def processFolder(base_dir, v, format, fstop = 0.0, bVideo = False, bImages = True, bCentral = 'Central'):
    base_dir2 = os.path.join(base_dir, v)
    total_names = [f for f in os.listdir(base_dir2) if f.endswith(format)]

    if len(total_names) == 0:
        return

    output_folder_str = base_dir2 + '_sdr'
    mkdir_s(output_folder_str)

    total_names = sorted(total_names)

    flag = True
    pML = -1.0;
    
    data_fn = os.path.join(base_dir, v + '_exposure.csv')
    
    if os.path.isfile(data_fn):
        os.remove(data_fn)
    
    data_f = open(data_fn, 'a+')
    data_f.write('Frame,Exposure\n')

    n = len(total_names) * 10
    tot = len(total_names)
    lst_to_write = []
    lst_data = []
    lst_data_f = []
    shift_exp = np.power(2.0, fstop)
    for i in range(0, tot - 1):
        name = total_names[i]
        name_dir = os.path.join(base_dir2, name)
        print(name_dir)
        img = npReadHDRWithCV(name_dir, False)

        pML, exp_i = computeAutomaticExposure(img, pML, 1.0 / 15.0, bCentral)

        width, height, cols  = img.shape
        if (flag and bVideo):
            flag = False
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(os.path.join(base_dir, v + '_fstop_' + str(fstop)  + '.mp4'), fourcc, 30.0, (width, height))

        exposure = exp_i * shift_exp

        if bImages:
            frame_gamma = npApplyGamma(img, exposure, 2.2)
            frame = fromNPtoVideoFrame(frame_gamma, -1.0, False)

            img_fn = os.path.splitext(name)[0] + '.png'
            name_dir = os.path.join(output_folder_str, img_fn)
            cv2.imwrite(name_dir, frame)

            if bVideo:
                out.write(frame)

        to_write = os.path.join(v, name) + ',' + str(exposure) + '\n'
        data_f.write(to_write)
            
    data_f.close()
    
    if bVideo:
        out.release()

if __name__ == '__main__':
    base_dir = sys.argv[1]
    format = sys.argv[2]

    video_folders_tmp = os.listdir(base_dir)
    video_folders_tmp = sorted(video_folders_tmp)

    bRun = True
    
    for i in range(0, len(video_folders_tmp)):
        v = video_folders_tmp[i]
        
        if(v.startswith(".")):
            continue
            
        full_path_folder = os.path.join(base_dir, v)
        
        if(os.path.isfile(full_path_folder)):
            continue

        print(v)
        processFolder(base_dir, v, format, 0.0, False, True)
