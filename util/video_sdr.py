#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import sys
import numpy as np
import cv2

from util.util_io import fromVideoFrameToNP, npImgRead

def videoIdentity(frame):
    return frame

class VideoSDR:

    #
    #
    #
    def __init__(self, video_path, video_fmt, f_frame = videoIdentity):
        self.video_path = video_path
               
        self.video_fmt = video_fmt
        self.f_frame = f_frame
        self.total_names = []
        self.v = []
        
        self.bVideo = not(os.path.isdir(self.video_path))
   
        self.counter = 0
        
        if self.bVideo:
            self.v = cv2.VideoCapture(self.video_path)
            self.n = int(self.v.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.total_names = sorted([f for f in os.listdir(video_path) if f.lower().endswith(video_fmt)])
            self.n = len(self.total_names)

            if self.n == 0:
                self.total_names = sorted([f for f in os.listdir(video_path) if f.lower().endswith('.jpg')])
                self.n = len(self.total_names)

            if self.n == 0:
                self.total_names = sorted([f for f in os.listdir(video_path) if f.lower().endswith('.png')])
                self.n = len(self.total_names)

            if self.n == 0:
                self.total_names = sorted([f for f in os.listdir(video_path) if f.lower().endswith('.jpeg')])
                self.n = len(self.total_names)

            if self.n == 0:
                self.total_names = sorted([f for f in os.listdir(video_path) if f.lower().endswith('.ppm')])
                self.n = len(self.total_names)

            if self.n == 0:
                print(video_path + ' folder does not contain either .jpg, .jpeg, .png, or .ppm files.')
     
    #
    #
    #
    def release(self):
        if self.v != []:
            self.v.release()
     
    #
    #
    #
    def getNumFrames(self):
        return self.n

    #
    #
    #
    def setFrame(self, frame):
        if self.bVideo:
            self.v.set(cv2.CAP_PROP_POS_FRAMES, frame)
            
        self.counter = frame
        
    #
    #
    #
    def getNextFrame(self, bBGR = True, index = -1):
        fn = ''
        if self.bVideo:
            success, frame_cv = self.v.read()
            if success:
                frame = fromVideoFrameToNP(frame_cv, bBGR)
            else:
                frame = []
        else:
            if index >= 0:
                self.counter = index
            
            fn = os.path.join(self.video_path, self.total_names[self.counter])
            frame = npImgRead(fn)
            self.counter = (self.counter + 1) % self.n
            success = True
        
        return success, fn, frame

    #
    #
    #
    def getNextFrameWithNext(self, bBGR = True, index = -1):
        fn = ''
        fnn = ''
        if self.bVideo:
            success, frame_cv = self.v.read()
            if success:
                frame = fromVideoFrameToNP(frame_cv, bBGR)
            else:
                frame = []
        else:
        
            if index >= 0:
                self.counter = index
            
            fn = os.path.join(self.video_path, self.total_names[self.counter])
            frame = npImgRead(fn)
            self.counter = (self.counter + 1) % self.n

            fnn = os.path.join(self.video_path, self.total_names[self.counter])

            success = True
        
        return success, fn, fnn, frame
