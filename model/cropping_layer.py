#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import torch
import copy
from torch import nn

#
#
#
def _get_crop_shape(target, refer):
    # print target.get_shape()[2], refer.get_shape()[2]
    # width, the 3rd dimension
    cw = (target.shape[3] - refer.shape[3])
    assert (cw >= 0), cw
    if cw % 2 != 0:
        cw1, cw2 = cw // 2, (cw // 2) + 1
    else:
        cw1, cw2 = cw // 2, cw // 2
    # height, the 2nd dimension
    ch = (target.shape[2] - refer.shape[2])
    assert (ch >= 0), ch
    if ch % 2 != 0:
        ch1, ch2 = ch // 2, (ch // 2) + 1
    else:
        ch1, ch2 = ch // 2, ch // 2
    
    return (ch1, ch2), (cw1, cw2)

#
#
#
def Cropping2D(ch, cw, t):
    s = t.shape
    return t[:,:,ch[0]:(s[2]-ch[1]), cw[0]:(s[3]-cw[1])]

#
#
#
def Crop2D(target, ref):
    ch, cw  = _get_crop_shape(target, ref)
    return Cropping2D(ch, cw, target)
