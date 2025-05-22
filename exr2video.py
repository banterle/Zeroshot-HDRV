#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#Main programmer: Francesco Banterle
#

import os
import sys

import cv2
import argparse

from util.util_io import fromNPtoVideoFrame, npReadHDRWithCV, createVideo
from util.util_np import npApplyGamma
 
#
#
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prediction Zeroshot-HDRV', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('exr_path', type=str, help='Path to data a folder with .exr files')
    parser.add_argument('--name_video', type=str, default ='output.mp4', help='name of the video')
    parser.add_argument('--fstop', type=float, default=0.0, help='f-stop value')
    parser.add_argument('--tmo', type=int, default=0, help='Tone mapping? (1)')

    args = parser.parse_args()
    
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

    #
    #
    #
    total_names = sorted([f for f in os.listdir(args.exr_path) if f.endswith('.exr')])
    n = len(total_names)

    for i in range(0, n):

        fn_i = os.path.join(args.exr_path, total_names[i])
        data = npReadHDRWithCV(fn_i)
        if i == 0:
            data_shape = data.shape
            print(data_shape)
            video = createVideo(args.name_video, data_shape[1], data_shape[0])

        if args.tmo != 1:
            data = npApplyGamma(data, args.fstop, 2.2, True)
        else:
            data = data / (data + 1.0) #global TMO
            data = npApplyGamma(data, 0.0, 2.2, True)
        data_out = fromNPtoVideoFrame(data, False, True)
        video.write(data_out)

    video.release()
