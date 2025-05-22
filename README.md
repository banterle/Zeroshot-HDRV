Zeroshot-HDRV (ZHDRV)
=============
![HDR-VDP](images/teaser.png?raw=true "Zeroshot-HDRV")

Zeroshot-HDRV is the official implementation of the paper [Self-supervised High Dynamic Range Imaging: What Can Be Learned from a Single 8-bit Video?](https://doi.org/10.1145/3648570) With Zeroshot-HDRV, a self-supervised approach, we show that, in many cases, a single SDR video is sufficient to generate an HDR video of the same quality or better than other state-of-the-art methods.

ADVANTAGES:
============
Since our method is self-supervised, it does not require external dataset during the learning phase. Furthermore, if a repeated pattern is well-exposed in some parts of the videos and this pattern is over-exposed in other parts, the method can learn this pattern and use this knowledge to reconstruct the missing pattern in over-exposed areas. Please see the image below:
![HDR-VDP](images/example.png?raw=true "Zeroshot-HDRV")

Note that the state-of-the-art method, since they are based on static weights that cannot adapt to the video cannot fully reconstruct the missing pattern. Please see the zoom of the previous image here:
![HDR-VDP](images/zoom.png?raw=true "Zeroshot-HDRV")


SETUP:
==============

Clone our repo and install dependencies. This is necessary for running ZHDRV.

```
conda create -n zhdrv
conda activate zhdrv
conda install ffmpeg
pip install -r requirements.txt
```

FFMPEG is required to process .mp4/.mov files, please follow [the instruction for instaling it](https://ffmpeg.org/download.html).
Note that if the video is already exported as a sequence of .png files, FFMPEG is not required.

HOW TO RUN IT:
==============
To run Zeroshot-HDRV, you need to launch the file ```zhdrv.py```. Some examples:

Testing Zeroshot-HDRV on a video composed by a series of .png files:

```
python3 zhdrv.py ./video01_sdr/ --data_type video
```

Testing Zeroshot-HDRV on a .mp4 video (this requires FFMPEG installed):

```
python3 zhdrv.py video01.mp4 --data_type video
```


OUTPUTS:
==============

When running ZHDRV on a video; e.g., fireplace.mp4, a hierarchy sof outputs will be generated in the folder ./runs; here is an example:

```
run
|
|___fireplace_lr0.0001_e128_b1_m4_t1_s-2
    |
    |___ckpt
    |
    |___fireplace_exr
    |
    |___recs
    |
    |___logs.csv
    |
    |___params.csv
```

the folder ```ckpt``` contains the best and the last weights checkpoints for the network, 
the folder ```fireplace_exr``` containts the EXR final frames of the videos after expansion, and
the folder ```recs``` containts the exposures results for the most over-exposed frame; this folder is updated during training and it is useful to
check what the network is learning. Finally, params.csv contains the paramters of the training and logs.csv the total loss function per epoch.


NOTES ON TIMINGS:
==============
This version of the code is not optimized for speed as the version used for the paper. This version is optimized to maximize compatibility with hardware (it does not require a CUDA-device) and video resolutions. Please refer to the paper timings.

WEIGHTS DOWNLOAD:
=================
There are no weights, this method is completely self-supervised.

REFERENCE:
==========

If you use Zeroshot-HDRV in your work, please cite it using this reference:

```
@article{10.1145/3648570,
author = {Banterle, Francesco and Marnerides, Demetris and Bashford-rogers, Thomas and Debattista, Kurt},
title = {Self-supervised High Dynamic Range Imaging: What Can Be Learned from a Single 8-bit Video?},
year = {2024},
issue_date = {April 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {43},
number = {2},
issn = {0730-0301},
url = {https://doi.org/10.1145/3648570},
doi = {10.1145/3648570},
journal = {ACM Trans. Graph.},
month = {mar},
articleno = {24},
numpages = {16},
keywords = {High dynamic range imaging, inverse tone mapping, deep learning, computational photography}
}
```
