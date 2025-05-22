#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import torch.nn.functional as F

#
#
#
def lossColor(img_rec, img, alpha = 5):
    loss_col = 1.0 - F.cosine_similarity(img_rec, img, dim = 1, eps = 1e-20)
    return alpha * loss_col.mean()

#
#
#
def lossL1C(img_rec, img):
    loss_sig_asia = F.l1_loss(img_rec, img) + 4 * lossColor(img_rec, img)
    return loss_sig_asia
