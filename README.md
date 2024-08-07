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


DEPENDENCIES:
==============

Requires the PyTorch library along with PIL, NumPy, SciPy, etc.

As the first step, you need to follow the [instructions for installing PyTorch](http://pytorch.org/).

To install dependencies, please use the following command: 

```bash
pip install -r requirements.txt 
```

FFMPEG is required to process .mp4 files, please follow [the instruction for instaling it](https://ffmpeg.org/download.html).
If the video is already exported as a sequence of .png files, FFMPEG is not required.

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

NOTES ON TIMINGS:
==============
This version of the code is not optimized for speed as the version used for the paper. This version is optimized to maximize compatibility with hardware (it does not require a CUDA-device), please refer to the paper timings.

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
