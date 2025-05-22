#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import numpy as np

#
#
#
class Distribution1D():
    #
    #
    #
    def __init__(self, x, bSorted = True):
        #build the PDF from the distribution x
        self.x_np = np.array(x)

        self.bSorted = bSorted
        
        if bSorted:
            self.indices = np.argsort(self.x_np)
            self.x_np = np.take_along_axis(self.x_np, self.indices, axis = 0)
            
        self.pdf = self.x_np
        sum_pdf = np.sum(self.pdf)
        
        n = self.x_np.shape[0]

        if sum_pdf > 0.0: #normalize the PDF
            self.pdf = self.pdf / sum_pdf
        else:             #uniform PDF
            self.pdf = np.ones(n) / n

        #compute the CDF from the PDF
        self.cdf = np.zeros(n)
        self.cdf[0] = self.x_np[0] / n
        for i in range(1, n):
            self.cdf[i] = self.cdf[i - 1] + (self.x_np[i] / n)
                
        if self.cdf[n - 1] > 0.0: #check to be sure to avoid singularities
            self.cdf /= self.cdf[n - 1]
            #print(self.cdf)

    #
    #
    #
    def sample(self, t):
        #check if t is in [0,1]
        if (t > 1.0) or (t < 0.0):
            return -1, -1.0
            
        j = np.searchsorted(self.cdf, t)
        return j, self.pdf[j]
    
    #
    #
    #
    def sampleRangeWithLimit(self, n, a = 0.0, b = 1.0, a_limit = 0.0):
        out = []
        
        #print(self.x_np)
        if a_limit > 0.0:
            index = np.where(self.x_np < a_limit)
            if len(index[0]) > 0:
                j = np.max(index[0])
                a = self.cdf[j]
                
        d_ab = b - a
        if d_ab > 0.0:
            dn = float(n - 1)
            for j in range(0, n):
                t = float(j) / dn
                t_s = t * d_ab + a
                j, pdf = self.sample(t_s)
                if self.bSorted:
                    j = self.indices[j]
                    
                #print([index, t_s,pdf])
                out.append(j)

        return out

if __name__ == '__main__':
    array = [33, 2, 2, 8, 2, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 1, 1, 8, 1, 2]
    print(array)
    c = Distribution1D(array, True)

    n = 4
    indices = c.sampleRangeWithLimit(n, 0.0, 1.0, 0.0)
    for i in range(0, n):
        j = indices[i]
        print([j, array[j]])
