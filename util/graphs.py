#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import matplotlib.pyplot as plt
import numpy as np

#
#
#
def plotGraph(array1, array2 = [], array3 = [], folder = '.', plot_name = 'plot.png'):
    # plot
    fig = plt.figure(figsize=(10, 4))
    n = min(len(array1), len(array2))
    plt.yscale("log")
    plt.plot(np.arange(1, n + 1), array1[0:n])  # train loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array2[0:n])  # train loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array3[0:n])  # train loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'val', 'test'], loc="upper left")
    title = os.path.join(folder, plot_name)
    plt.savefig(title, dpi=600)
    plt.close(fig)
    
#
#
#
def plotGraphSingle(array1, folder = '.', legend_str = 'train', plot_name = 'plot.png'):
    # plot
    fig = plt.figure(figsize=(10, 4))
    n = len(array1)
    plt.plot(np.arange(1, n + 1), array1[0:n])  # train loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend([legend_str], loc="upper left")
    title = os.path.join(folder, plot_name)
    plt.savefig(title, dpi=600)
    plt.close(fig)
