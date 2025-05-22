#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import numpy as np
from PIL import Image

#
#
#
def crop_center(img,csize):
    y,x = img.shape
    cx = x // 2
    cy = y // 2
    out = img[(cy-csize):(cy+csize),(cx-csize):(cx+csize)]
    return out

#
#
#
def pilDataAugmentation(img, j):
    img_out = []
    if(j == 0):
        img_out = img
    elif (j == 1):
        img_out = img.rotate(90)
    elif (j == 2):
        img_out = img.rotate(180)
    elif (j == 3):
        img_out = img.rotate(270)
    elif (j == 4):
        img_out = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    elif (j == 5):
        img_out = img.rotate(90)
        img_out = img_out.transpose(method=Image.FLIP_LEFT_RIGHT)
    else:
        img_out = img.transpose(method=Image.FLIP_TOP_BOTTOM)
    return img_out
