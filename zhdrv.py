#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import sys
import argparse
import subprocess

from video2png import getColorSpaceInfo

#
#
#
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Zeroshot-HDRV', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_path', type=str, help='Path to data dir or to a movie (.mov/.mp4) or a folder with images (.png)')
    parser.add_argument('--data_type', type=str, default ='video', help='are we working with videos (\'video\') or static images (\'image\')')
    parser.add_argument('--already_trained', type=str, default ='False', help='was the network already trained?')
    parser.add_argument('--n_exp_down', type=int, default = 2, help='the number of exposures to generate down (under-exposed)')
    parser.add_argument('--n_exp_up', type=int, default = 2, help='the number of exposures to generate up (over-exposed)')

    args = parser.parse_args()

    data_path = args.data_path
  
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

    name = os.path.basename(os.path.normpath(data_path))
    
    if os.path.isfile(data_path):
        name = os.path.splitext(name)[0]

    params = vars(args)
    
    #
    #this data is from the paper
    #
    params['epochs'] = 128
    params['batch'] = 1
    params['mode'] = 4
    params['batch'] = 1
    params['lr'] = 0.0001
    params['sampling'] = -2
    params['temp'] = 1
    
    if args.data_type == 'image':
        params['sampling'] = 1
        params['temp'] = 0

    print(args.already_trained)
    if args.already_trained == 'False':
        if args.data_type == 'video':
            #is this video a folder or a .mp4 file?
            video_str = os.path.splitext(data_path)[1].lower()
            if video_str == '.mp4' or video_str == '.mov':
                print("Converting video into images...")
                subprocess.call('python video2png.py ' + data_path, shell=True)
                    
            data_path = os.path.splitext(data_path)[0]
            subprocess.call('python train.py ' + data_path + ' --name ' + name + ' --epoch ' + str(params['epochs']) + ' --format .png', shell=True)
            params['temp'] = 1
        elif args.data_type == 'image':
            subprocess.call('python train.py ' + data_path + ' --name ' + name + ' --sampling 1 --temp 0' + ' --format .png', shell=True)
            params['temp'] = 0
            params['sampling'] = 1

    lin_fun = 'sRGB'
    
    if args.data_type == 'video':
        result = getColorSpaceInfo(data_path)
        if ('bt470bg' in result) or ('unknown' in result):
            lin_fun = 'gamma_2.2'

    run_name = name + '_lr{0[lr]}_e{0[epochs]}_b{0[batch]}_m{0[mode]}_t{0[temp]}_s{0[sampling]}'.format(params)
    run_name = os.path.join('runs', run_name)

    print(run_name)
    
    output_path = os.path.join(run_name, name + '_exr')
    
    exposures_str = ' --n_exp_down ' + str(args.n_exp_down) + ' --n_exp_up ' + str(args.n_exp_up)
    subprocess.call('python pred.py ' + data_path + ' --model_path ' + run_name + ' --output_path ' + output_path + ' --transferfunction ' + lin_fun + exposures_str, shell=True)
