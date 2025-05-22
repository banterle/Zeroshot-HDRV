#
#Copyright (C) 2020-2024 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#
#
#Main programmer: Francesco Banterle
#

import os
import numpy as np
from PIL import Image
import PIL.ExifTags

#
#
#
def fromPILtoNP(img):
    img_np = np.array(img).astype('float') / 255.0
    return img_np

#
#
#
def fromNPtoPIL(img):
    formatted = (img * 255).astype('uint8')
    img_pil = Image.fromarray(formatted)
    return img_pil

#
# img ==> h x w x c
#
def lum(img, bBGR=False):
    if bBGR:
        m = (2, 1, 0)
    else:
        m = (0, 1, 2)

    sz = img.shape

    if sz[2] == 3:
        L = 0.213 * img[:, :, m[0]] + 0.715 * img[:, :, m[1]] + 0.072 * img[:, :, m[2]]
    else:
        L = np.mean(img, axis=2)

    return L

#
#
#
def estimateAverageLuminance(
    exposure_time, aperture_value=1.0, iso_value=1.0, K_value=12.5
):
    K_value = np.clip(K_value, 10.6, 13.4)
    value = (K_value * aperture_value * aperture_value) / (iso_value * exposure_time)
    value_inv = (iso_value * exposure_time) / (
        K_value * aperture_value * aperture_value
    )
    return value, value_inv

#
#
#
def weightFunction(
    img, weight_type='Deb97', b_mean_weight=True, bounds=(0.01, 0.99), pp=[], bBGR=True
):

    sz = img.shape
    if b_mean_weight:
        x = np.zeros(sz)
        L = lum(img)
        for i in range(0, sz[2]):
            x[:, :, i] = L
    else:
        x = img

    bWeight = True

    if weight_type == 'identity':
        weight = x
        bWeight = False

    if weight_type == 'reverse':
        weight = 1.0 - x
        bWeight = False

    if weight_type == 'hat':
        tmp = 2.0 * x - 1.0
        weight = 1.0 - np.power(tmp, 12.0)
        bWeight = False

    if weight_type == 'Deb97':
        Z_min = bounds[0]
        Z_max = bounds[1]
        Z_mid = (Z_min + Z_max) / 2.0
        delta = Z_max - Z_min

        weight = np.zeros(sz)

        ind1 = np.where(x <= Z_mid)
        ind2 = np.where(x > Z_mid)
        weight[ind1] = x[ind1] - Z_min
        weight[ind2] = Z_max - x[ind2]

        if delta > 0.0:
            weight /= delta

        weight = weight.clip(0.0, 1.0)
        bWeight = False

    if bWeight:
        weight = np.ones(sz)

    return weight

#
#
#
def removeSRGB(x):
    if x <= 0.04045:
        return x / 12.92
    else:
        y = (x + 0.055) / (1.055)
        return np.power(y, 2.4)

#
#
#
def removeCRF(img, lin_type='gamma', lin_fun=2.2):

    if lin_type == 'gamma':
        img = np.power(img, lin_fun)
        
    if lin_type == 'sRGB':
        img = np.vectorize(removeSRGB)(img)

    return img

#
#
#
def buildHDR(
    stack,
    stack_exposure,
    lin_type,
    lin_fun,
    weight_type,
    merge_type='log',
    b_mean_weight=True,
    bBGR=True,
):
    n_i = len(stack)
    n_e = len(stack_exposure)

    if n_i != n_e:
        return []

    e_check = set(stack_exposure)

    if len(e_check) != n_e:
        return []

    delta_value = 1.0 / 65535.0

    sz = stack[0].shape
    imgOut = np.zeros(sz)
    tot_w = np.zeros(sz)

    # check saturated pixels
    e_min = min(stack_exposure)
    index_saturated = stack_exposure.index(e_min)

    # check noisy pixels
    e_max = max(stack_exposure)
    index_noisy = stack_exposure.index(e_max)

    for i in range(0, n_i):
        img_i = stack[i]

        if img_i.dtype == 'uint8':
            img_i = img_i.astype('float32')
            img_i /= 255.0

        if img_i.dtype == 'uint16':
            img_i = img_i.astype('float32')
            img_i /= 65535.0

        weight_type_i = weight_type
        
        if i == index_saturated:
            # does this image have saturated pixels?
            index = np.where(img_i > 0.9)
            if len(index[0]) > 0:
                weight_type_i = 'identity'

        if i == index_noisy:
            # does this image have noisy pixels?
            index = np.where(img_i < 0.1)
            if len(index[0]) > 0:
                weight_type_i = 'reverse'

        w_i = weightFunction(
            img_i, weight_type_i, b_mean_weight, (0.01, 0.99), [], bBGR
        )

        dt_i = stack_exposure[i]
        img_i = removeCRF(img_i, lin_type, lin_fun)

        bMerge = True

        if merge_type == 'log':
            imgOut += w_i * (np.log(img_i + delta_value) - np.log(dt_i))
            tot_w += w_i
            bMerge = False

        if merge_type == 'w_time_sq':
            imgOut += w_i * img_i * dt_i
            tot_w += w_i * dt_i * dt_i
            bMerge = False

        # linear merge
        if bMerge:
            imgOut += w_i * img_i / dt_i
            tot_w += w_i

    imgOut /= tot_w

    if merge_type == 'log':
        imgOut = np.exp(imgOut)

    return imgOut

#
#
#
def buildHDRwithPIL(folder_name, extension):
    total_names = [f for f in os.listdir(folder_name) if f.endswith('.' + extension)]
    total_names.sort()
    stack = []
    exposures = []

    for name in total_names:
        img_pil = Image.open(os.path.join(folder_name, name))

        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in img_pil._getexif().items()
            if k in PIL.ExifTags.TAGS
        }

        if 'ExposureTime' in exif.keys():
            exposure_time = exif['ExposureTime']
            et = exposure_time[0] / exposure_time[1]
        else:
            exposure_time = 1.0

        if 'FNumber' in exif.keys():
            aperture = exif['FNumber']
            a = aperture[0] / aperture[1]
        else:
            a = 1.0

        if 'ISOSpeedRatings' in exif.keys():
            ISO = exif['ISOSpeedRatings']
        else:
            ISO = 1.0

        v, v_inv = estimateAverageLuminance(et, a, ISO)

        img_np = fromPILtoNP(img_pil)
        stack.append(img_np)
        exposures.append(v_inv)

    out = buildHDR(stack, exposures, 'gamma', 2.2, 'Deb97', 'log', True, False)
    return out


if __name__ == "__main__":

    print(removeCRF(np.ones((4,4,3))*0.5, lin_type = 'sRGB'))
    #out = buildHDRwithPIL('stack', 'jpg')
    #tonemap = np.power(out / (out + 1.0), 1.0 / 2.2)
    #tonemap = np.clip(tonemap, a_min=0.0, a_max=1.0)
    #tonemap_pil = fromNPtoPIL(tonemap)
    #tonemap_pil.save('test.png')
