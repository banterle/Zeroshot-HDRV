#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import torch
from torch import nn

#
#
#
class UberPool(nn.Module):

    #
    #
    #
    def __init__(self):
        super().__init__()
        self.max2d = nn.MaxPool2d((2,2), stride =(2,2))
        self.avg2d = nn.AvgPool2d((2,2), stride =(2,2))
        
        self.alpha = torch.autograd.Variable(torch.randn(1).type(torch.FloatTensor), requires_grad=True)
        
        if torch.cuda.is_available():
            self.alpha = self.alpha.cuda()

    #
    #
    #
    def forward(self, input):
        x0 = self.max2d(input)
        x1 = self.avg2d(input)
        return x0 * self.alpha + (1.0 - self.alpha) * x1
