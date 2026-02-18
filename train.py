#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import time
import sys
import gc
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util.graphs import *
from util.io import *
from util.util_np import *
from util.util_io import *
from util.dataset import *
from util.create_dataset_video_sdr import *
from model.model_ud import *
from model.loss import *
from util.util_torch import getImage2Tensor

#
#
#
def train(epoch, loader, model, optimizer, args, scheduler = None):
    model.train()
    torch.autograd.set_detect_anomaly(True)
        
    progress = tqdm(loader)
    
    total_loss = 0.0
    total_rec = 0.0
    total_temporal = 0.0
    counter = 0
    
    optimizer.zero_grad()
  
    #mode 2 and 4
    #gc.collect()

    for f0, o0, o0_n, f0_n in progress:

        if torch.cuda.is_available():
            f0 = f0.cuda()
            o0 = o0.cuda()

            if (args.temp > 0):
                o0_n = o0_n.cuda()
                f0_n = f0_n.cuda()
        
        if args.mode == 0:
            #ACM SIGGRAPH 2021 and ACM SIGGRAPH ASIA 2021 submissions
            f0_d = model.fD(f0)
            o0_u = model.fU(o0)
                    
            loss_rec = lossL1C(o0, f0_d) + lossL1C(f0, o0_u)
        elif args.mode == 1:
            #ICCP 2022 submission: DDUU --> model = UNetUD(3, 3, False, args.es, 1)
            f0_d = model.fD(f0)
            o0_u = model.fU(o0)

            loss_rec = lossL1C(o0, f0_d) + lossL1C(f0, o0_u)
            o0_du = model.fU(model.fD(o0))
            #o0_dduu = model.fU((model.fU(model.fD(model.fD(o0)))))
            loss_rec += lossL1C(o0_du, o0) * 0.25
        elif (args.mode == 2) or (args.mode == 4):
            #IEEE CVPR 2022 Submission: delta mul --> model = UNetUD(3, 3, False, args.es, 1, None, False)
            delta = o0 / (f0 + model.min_val)
            delta_p = model.fD(f0)
                        
            if args.diff == 0:
                f0_d = delta_p * f0
                loss_d = F.mse_loss(delta_p, delta)
            else:
                f0_d = torch.clamp(delta_p * f0, 0.0, 1.0)
                loss_d = F.l1_loss(delta_p, delta)
                
            loss_r0 = lossL1C(f0_d, o0)
            loss_rec = loss_d * 4.0 + loss_r0
            
            if args.mode == 4: #ACM TOG submission
                o0_u = o0 / (delta_p + model.min_val)
                o0_u = torch.clamp(o0_u, 0.0, 1.0)
                loss_r1 = lossL1C(o0_u, f0)
                loss_rec += 0.5 * loss_r1

        #total loss
        total_rec += loss_rec
        
        #
        #temporal loss
        #
        loss_t = 0.0
            
        if (args.temp == 1):#L_Jacobian
            f0_n_d = model.getExpD(f0_n)

            d_1 = (o0_n - o0)
            d_2 = (f0_n_d - f0_d)
            loss_t = F.l1_loss(d_1, d_2)
            
        if (args.temp == 2):#L_Stability
            f0_n_d = model.getExpD(f0_n)
            loss_t = F.mse_loss(f0_n_d, f0_d)

        if (args.temp > 0):
            loss = loss_rec * (1.0 - args.alpha) + loss_t * args.alpha
        else:
            loss = loss_rec

        total_temporal += loss_t

        #
        #final loss
        #
        loss.backward()
        
        if args.batch > 0.0:
            optimizer.step()
            optimizer.zero_grad()
        else:
            batchSize = -args.batch
            if(counter % batchSize == 0):
                optimizer.step()
                optimizer.zero_grad()
                print("Gradient Update")
                
        total_loss += loss.item()
        counter += 1

        progress.set_postfix({'loss': total_loss / counter})

    avg_loss = total_loss / counter
    avg_rec = total_rec.item() / counter
    
    if args.temp > 0:
        avg_temporal = total_temporal.item() / counter
    else:
        avg_temporal = 0.0
    return avg_loss, avg_rec, avg_temporal

#
# the main program
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train  Zeroshot-HDRV', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=str, help='Path to data dir')
    parser.add_argument('--name', type=str, default ='hdrv', help='Name of the training')
    parser.add_argument('-g', '--group', default = 7, type=int, help='grouping factor for augmented dataset')
    parser.add_argument('-e', '--epochs', type=int, default=128, help='Number of training epochs')
    parser.add_argument('-b', '--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--es', type=float, default=2.0, help='Exposure Shift')
    parser.add_argument('--alpha', type=float, default=0.95, help='Alpha')
    parser.add_argument('-s', '--sampling', type=int, default=-2, help='Sampling rate for the frames of the video. -2: Uniform sampling based on well-exposedness as used in the paper.')
    parser.add_argument('--ensemble', type=int, default=0, help='Ensemble')
    parser.add_argument('--format', type=str, default='.png', help='format of the data if image files')
    parser.add_argument('--resume', type=str, default='', help='Shall we resume?')
    parser.add_argument('-m', '--mode', type=int, default=4, help='Mode')
    parser.add_argument('-t', '--temp', type=int, default=1, help='Temporal Loss')
    parser.add_argument('-d', '--diff', type=int, default=0, help='Differences Loss')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale values of the input frames')
    parser.add_argument('--samples_is', type=int, default=128, help='Samples')
    parser.add_argument('-r', '--runs', type=str, default='./runs', help='Base dir for runs')
    parser.add_argument('--debug', type=str, default='yes', help='Debugging mode')

    args = parser.parse_args()

    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

    if torch.cuda.is_available():
        print("CUDA is On")
    else:
        print("CUDA is off")

    print('F-stop: ' + str(args.es))
    print('Alpha: ' + str(args.alpha))
    
    if args.resume == '':
        print('Resume? No')
    else:
        print('Resume? ' + str(args.resume))
        
    print('Mode: ' + str(args.mode))
    print('Temporal Coherency: ' + str(args.temp))
    print('Ensemble: ' + str(args.ensemble))
    print('Scaling factor: ' + str(args.scale))
    print('Sampling: ' + str(args.sampling))
    print('Samples: ' + str(args.samples_is))
    print('Diff Loss: ' + str(args.diff))
    
    bDebug = (args.debug == 'yes')

    ### Prepare run dir
    params = vars(args)

    if (args.ensemble == 1) and (args.name == 'hdrv'):
        args.name = os.path.basename(args.data)

        if args.name == '':
            args.name = 'hdrv_ensemble'

    run_name = args.name + '_lr{0[lr]}_e{0[epochs]}_b{0[batch]}_m{0[mode]}_t{0[temp]}_s{0[sampling]}'.format(params)

    mkdir_s(args.runs)

    run_dir = os.path.join(args.runs, run_name)
    ckpt_dir = os.path.join(run_dir, 'ckpt')
    recs_dir = os.path.join(run_dir, 'recs')

    print(run_dir)
    mkdir_s(run_dir)
    mkdir_s(ckpt_dir)
    mkdir_s(recs_dir)
        
    args.recs_dir = recs_dir;
    
    log_file = os.path.join(run_dir, 'log.csv')
    param_file = os.path.join(run_dir, 'params.csv')
    pd.DataFrame(params, index=[0]).to_csv(param_file, index=False)

    args_data = args.data
    if args.ensemble == 1:
        #training multiple videos at the same time
        train_data, filename_rec = genDataset(args.data, args)
    else:
        args.data = getGlobalPath(args.data)
        train_data, filename_rec, num_frames = split_data_from_video_sdr(args_data, args.es, group=args.group, sampling = args.sampling, recs_dir = args.recs_dir, scaling = args.scale, samples_is = args.samples_is, format = args.format)

    bTypeRec = isinstance(filename_rec, list)

    #representative image (most over-exposed one)
    if bTypeRec:
        img = npImgRead(filename_rec[0])
    else:
        img = npImgRead(filename_rec)

    num_pixels = img.shape[0] * img.shape[1]

    bTemporal = (args.temp > 0)
    train_data = SDRDataset(train_data, group = args.group, expo_shift = args.es, scale = args.scale, area = num_pixels, temporal = bTemporal)
    
    train_loader = DataLoader(train_data,  batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True, persistent_workers = True)

    #
    #create the model
    #
    
    #do we need to resume training?
    if args.resume != '':
        if args.resume == 'same':
            resume_str = run_dir
        else:
            resume_str = arg.resume
        print('Resume weights: ' + resume_str)
    else:
        resume_str = None
        
    n_input_val = 3
    n_output_val = 3

    model = UNetUD(n_input_val, n_output_val, args.es, resume_str, args.mode)

    if torch.cuda.is_available():
        model = model.cuda()
        
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    log = pd.DataFrame()

    ### Train loop
    best_mse = None
    ckpt_prev = ''
    cur_loss_vec = []
    rec_loss_vec = []
    temp_loss_vec = []

    if bTypeRec == False:
        name_rec = os.path.splitext(filename_rec)[0]    
        sz_sdr = img.shape
        img, bFlag = addBorder(img, 16)
        img_t = getImage2Tensor(img)

    start_epoch = 1

    if args.resume:
        start_epoch = model.epoch
        best_mse = model.best_mse
        print('Best MSE: ' + str(best_mse))

    for epoch in trange(start_epoch, args.epochs + 1):
        train_data.epoch = epoch
        cur_loss, rec_loss, temp_loss = train(epoch, train_loader, model, optimizer, args, scheduler)

        if bDebug:
            cur_loss_vec.append(cur_loss)
            rec_loss_vec.append(rec_loss)
            temp_loss_vec.append(temp_loss)

            metrics = {'mse': float(cur_loss)}
            metrics['epoch'] = int(epoch)

            log = log._append(metrics, ignore_index=True)
            log.to_csv(log_file, index=False)

        if (best_mse is None) or (cur_loss < best_mse) or (epoch == args.epochs):
            
            if bDebug:
                plotGraphSingle(cur_loss_vec, ckpt_dir, 'Loss', 'plot_loss_full.png')
                plotGraphSingle(rec_loss_vec, ckpt_dir, 'Loss', 'plot_loss_reconstruction.png')
                plotGraphSingle(temp_loss_vec, ckpt_dir, 'Loss', 'plot_loss_temporal.png')
            
            if bDebug:
                model.eval()

                if bTypeRec:
                    for name in filename_rec:
                        img = npImgRead(name)
                        name_rec = os.path.splitext(name)[0]    
                        sz_sdr = img.shape
                        img, bFlag = addBorder(img, 16)
                    
                        img_t = getImage2Tensor(img)
                        img_dd, img_d, img_u, img_uu, delta_img, delta_img_d, delta_img_u = model.predict4(img_t)

                        if bFlag:
                            img_dd = img_dd[:,0:sz_sdr[0],0:sz_sdr[1]]
                            img_d = img_d[:,0:sz_sdr[0],0:sz_sdr[1]]
                            img_u = img_u[:,0:sz_sdr[0],0:sz_sdr[1]]
                            img_uu = img_uu[:,0:sz_sdr[0],0:sz_sdr[1]]
                    
                        npSaveImage(img_dd,         name_rec + '_-4.png')
                        npSaveImage(img_d,          name_rec + '_-2.png')
                        npSaveImage(img_u,          name_rec + '_+2.png')
                        npSaveImage(img_uu,         name_rec + '_+4.png')

                        if not (delta_img == None):
                            npSaveImage(delta_img,      name_rec + '_delta_img.png')
                            npSaveImage(delta_img_d,    name_rec + '_delta_img_d.png')
                            npSaveImage(delta_img_u,    name_rec + '_delta_img_u.png')
                else:
                    img_dd, img_d, img_u, img_uu, delta_img, delta_img_d, delta_img_u = model.predict4(img_t)
                    if bFlag:
                        img_dd = img_dd[:,0:sz_sdr[0],0:sz_sdr[1]]
                        img_d = img_d[:,0:sz_sdr[0],0:sz_sdr[1]]
                        img_u = img_u[:,0:sz_sdr[0],0:sz_sdr[1]]
                        img_uu = img_uu[:,0:sz_sdr[0],0:sz_sdr[1]]
                    
                    npSaveImage(img_dd,      name_rec + '_-4.png')
                    npSaveImage(img_d,       name_rec + '_-2.png')
                    npSaveImage(img_u,       name_rec + '_+2.png')
                    npSaveImage(img_uu,      name_rec + '_+4.png')

                    if not (delta_img is None):
                        npSaveImage(delta_img,   name_rec + '_delta_img.png')
                        npSaveImage(delta_img_d, name_rec + '_delta_img_d.png')
                        npSaveImage(delta_img_u, name_rec + '_delta_img_u.png')

            best_mse = cur_loss
            ckpt = os.path.join(ckpt_dir, 'ckpt_e{}.pth'.format(epoch))
            torch.save({
                'n_input': n_input_val,
                'n_output': n_output_val,
                'epoch': epoch,
                'mode': args.mode,
                'es': args.es,
                'mse': best_mse,
                'scale': args.scale,
                'sampling': args.sampling,
                'temp': args.temp,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, ckpt)

            if ckpt_prev and (epoch < (args.epochs - 1)):
                if os.path.isfile(ckpt_prev):
                    os.remove(ckpt_prev)
                    
            ckpt_prev = ckpt

        scheduler.step(cur_loss)
        
