#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#
#
#
def save_distribution(frames_oe, output_path="frames_oe.png", bins=32, dpi=300):
    frames_oe = np.asarray(frames_oe)
    frames_oe = frames_oe[np.isfinite(frames_oe)]

    if frames_oe.size == 0:
        return

    fig, ax = plt.subplots()

    #compute histogram
    ax.hist(frames_oe, bins=bins, density=True, alpha=0.4, edgecolor="black")

    x = np.linspace(frames_oe.min(), frames_oe.max(), 500)
    n = frames_oe.size
    std = frames_oe.std(ddof=1)

    if n > 1 and std > 0:
        bandwidth = 1.06 * std * n ** (-1 / 5)

        kernels = np.exp(-0.5 * ((x[:, None] - frames_oe[None, :]) / bandwidth) ** 2)
        kde = kernels.sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))

        ax.plot(x, kde, linewidth=2)

    #plot
    _, ymax = ax.get_ylim()
    ax.vlines(frames_oe, 0, ymax * 0.03, linewidth=0.8)

    ax.set_xlabel("frames_oe")
    ax.set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

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
