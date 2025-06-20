#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import argparse
import numpy as np
from torchvision.transforms.functional import to_tensor
import cv2

from model.model_ud import *
from util.util_np import *
from util.util_io import *
from util.io import *
from util.dataset import *

from util.util_torch import getImage2Tensor
from util.merge_hdr import buildHDR
from util.video_sdr import VideoSDR
    
#
#
#
def buildHDRFromNet(model, frame_np, bDownOnly = False, transferfunction = 'gamma_2.2', n_exp_down = 2, n_exp_up = 2):
    frame_torch = getImage2Tensor(frame_np)
            
    #expo0 = np.power(2.0, -model.fstop)
    #expo1 = np.power(2.0, -model.fstop * 2)
    #expo3 = np.power(2.0,  model.fstop)
    #expo4 = np.power(2.0,  model.fstop * 2)
    #ft_dd, ft_d, ft_u, ft_uu = model.predict4(frame_torch)

    #if bDownOnly:
    #    stack_exposure = [expo0, expo1, 1.0]
    #    stack = [fromTorchToNP(ft_dd), fromTorchToNP(ft_d), frame_np]
    #else:
    #    stack_exposure = [expo0, expo1, 1.0, expo3, expo4]
    #    stack = [fromTorchToNP(ft_dd), fromTorchToNP(ft_d), frame_np, fromTorchToNP(ft_u), fromTorchToNP(ft_uu)]
    
    stack, stack_exposure = model.predict(frame_torch, n_exp_down, n_exp_up)

    if 'gamma_' in transferfunction:
        lin_type = 'gamma_'
        try:
            lin_fun = float(transferfunction.replace('gamma_', ' '))
            if lin_fun <= 0.0:
                lin_fun = 2.2
        except:
            lin_fun = 2.2
    
    if transferfunction == 'sRGB':
        lin_type = 'sRGB'
        lin_fun = []
        
    img_hdr_np = buildHDR(stack, stack_exposure, lin_type, lin_fun, 'hat', 'log', True, False)
    img_hdr_np = np.float32(img_hdr_np)
    sz = frame_torch.shape
    return img_hdr_np, sz

#
#
#
def evalFolder(args, bWrite = True, bVideo = False):

    video_path = args.video_path

    video_name = os.path.splitext(video_path)[0]
        
    print('Video path: ' + video_path)
    print('Weights path: ' + args.model_path)

    v = VideoSDR(video_path, '.png')
    
    if args.output_path == 'same':
        output_path = video_path
    else:
        output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = UNetUD(3, 3, 2.0, args.model_path)
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    model_suffix_s = model.getString()
    suffix = '_our' + model_suffix_s

    n = v.getNumFrames()
    video = None
    bDelta = False
   
    for i in range(0,n):
        if True:
            #get the current frame
            success, fn, frame = v.getNextFrame(True, i)
            
            sz_sdr = frame.shape
            #frame, bFlag = addBorder(frame, 16)
            
            #expand the frame and build an HDR frame
            img_hdr, sz = buildHDRFromNet(model, frame * args.scale, False, args.transferfunction, n_exp_down = args.n_exp_down, n_exp_up = args.n_exp_up)
            
            #if bFlag:
            #    img_hdr = img_hdr[0:sz_sdr[0],0:sz_sdr[1],:]
            
            sz = img_hdr.shape

            #output exr frames
            name_out = os.path.basename(fn)
            print(name_out)

            if bWrite:
                fn_hdr_out = os.path.splitext(name_out)[0] + suffix + '_' + str(i).zfill(6) +'.exr'
                name_img = os.path.join(output_path, fn_hdr_out)
                cv2.imwrite(name_img, switchBGR(img_hdr))
                               
            #output a tone mapped video
            if (i == 0) and bVideo:
                name_video = os.path.join(output_path, video_name + suffix + '.mp4')
                print('Video file: ' + name_video)
                print('Res: ' + str(sz[0]) + ' x ' + str(sz[1]))
                video = createVideo(name_video, sz[0] * 5 , sz[1])

            if (video != None):
                frame_out = frame_hdr;
                frame_out = fromNPtoVideoFrame(frame_out, -1.0, True)
                video.write(frame_out)
    
    #release the intput SDR video
    if v != None:
        v.release()
        
    #release the output video
    if video != None:
        video.release()

    del model

#
#
#
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prediction Zeroshot-HDRV', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video_path', type=str, help='Path to data dir or .mov/.mp4 file')
    parser.add_argument('--model_path', type=str, default ='', help='Path to data dir')
    parser.add_argument('--output_path', type=str, default='./', help='Output path')
    parser.add_argument('--scale', type=float, default=1.0, help='Scaling')
    parser.add_argument('--transferfunction', type=str, default='gamma_2.2', help='Transfer function')
    parser.add_argument('--n_exp_down', type=int, default = 2, help='the number of exposures to generate down (under-exposed)')
    parser.add_argument('--n_exp_up', type=int, default = 2, help='the number of exposures to generate up (over-exposed)')

    args = parser.parse_args()

    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

    evalFolder(args, True, False)
