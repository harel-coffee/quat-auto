#!/usr/bin/env python3
"""
Purely image based features
"""
"""
    This file is part of quat.
    quat is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    quat is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with quat. If not, see <http://www.gnu.org/licenses/>.

    Author: Steve Göring
"""
import json
import numpy as np
import pandas as pd
import cv2
import skimage.measure
import skvideo.measure
from skimage import img_as_ubyte
import skimage.color
from skimage import exposure



def color_fulness_features(image_rgb):
    """
    calculates color fullness

    re-implementated by Serge Molina

    References
    ----------
    - Hasler, David, and Sabine E. Suesstrunk. "Measuring colorfulness in natural images."
      In: Human vision and electronic imaging VIII. Vol. 5007. International Society for Optics and Photonics, 2003.
    """
    assert(len(image_rgb.shape) == 3)

    rg = (image_rgb[:, :, 0] - image_rgb[:, :, 1]).ravel()
    yb = (image_rgb[:, :, 0] / 2 + image_rgb[:, :, 1] / 2 - image_rgb[:, :, 2]).ravel()

    rg_std = np.std(rg)
    yb_std = np.std(yb)

    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)

    trigo_len_std = np.sqrt(rg_std ** 2 + yb_std ** 2)
    neutral_dist = np.sqrt(rg_mean ** 2 + yb_mean ** 2)

    return float(trigo_len_std + 0.3 * neutral_dist)


def calc_tone_features(image, gray=False):
    """
    calculate tone feature,

    re-implemented by Serge Molina

    References
    ----------
    - T. O. Aydın, A. Smolic, and M. Gross. "Automated aesthetic analysis of photographic images".
      In: IEEE transactions on visualization and computer graphics 21.1 (2015), pp. 31–42.
    """
    if not gray:
        image_gray = skimage.color.rgb2gray(image)
    else:
        image_gray = image

    image_1d = image_gray.ravel()
    percentile05_value = np.percentile(image_1d, 5)
    percentile95_value = np.percentile(image_1d, 95)

    percentile30_value = np.percentile(image_1d, 30)
    percentile70_value = np.percentile(image_1d, 70)

    u = 0.05
    o = 0.05

    c_u = min(u, percentile95_value - percentile70_value) / u
    c_o = min(o, percentile30_value - percentile05_value) / o

    return c_u * c_o * (percentile95_value - percentile05_value)


def calc_contrast_features(frame):
    """
    calculates contrast based on histogram equalization,

    based on julan zebelein's master thesis
    """
    frame = img_as_ubyte(frame)
    hist, bins = np.histogram(frame.flatten(), 1024, [0, 1024])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 1024 / (cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[frame]

    hist2, bins = np.histogram(img2.flatten(), 1024, [0, 1024])
    cdf2 = hist2.cumsum()
    cdf2_normalized = cdf2 * hist2.max() / cdf2.max()

    sumAverageDifCDF = 0
    for x in range(256):
        histValue = cdf_normalized[x]
        perfectHistValue = cdf2_normalized[x]

        histValuePercent = (100 * histValue) / perfectHistValue
        difPercent = abs(histValuePercent - 100)

        sumAverageDifCDF += difPercent

    avgDif = 100 - sumAverageDifCDF / len(cdf_normalized)
    return float(avgDif)


def calc_fft_features(frame, debug=False):
    """
    calculates fft feature,

    based on julan zebelein's master thesis

    References
    ----------
    - I. Katsavounidis et al. "Native resolution detection of video sequences".
      In: Annual Technical Conference and Exhibition, SMPTE 2015. SMPTE. 2015, pp. 1–20.
    """

    def radial_profile(data, center):
        y, x = np.indices((data.shape))
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(np.int)

        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile

    # start video
    file_width = int(frame.shape[1])
    file_height = int(frame.shape[0])
    frame = np.uint8(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prefinal = cv2.resize(gray, (file_width, file_height))
    #final = cv2.GaussianBlur(prefinal,(5,5),0)
    final = cv2.bilateralFilter(prefinal, 9, 75, 75)

    f = np.fft.fft2(final)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(0.00000001 + np.abs(fshift))

    file_height, file_width = magnitude_spectrum.shape
    CurrentCenter = (file_width / 2, file_height / 2)

    # calculate the azimuthally averaged 1D power spectrum
    psf1D = radial_profile(magnitude_spectrum, CurrentCenter)
    lowFreqInd = int((len(psf1D) / 2))

    psf1D_onlyHighFreq = psf1D[lowFreqInd:]
    sum_of_high_frequencies = sum(psf1D_onlyHighFreq)

    return float(sum_of_high_frequencies)


def calc_saturation_features(frame, debug=True):
    """
    calculates saturation of a given image,

    re-implemented by Serge Molina

    References
    ----------
    - T. O. Aydın, A. Smolic, and M. Gross. "Automated aesthetic analysis of photographic images".
      In: IEEE transactions on visualization and computer graphics 21.1 2015, pp. 31–42.""
    """
    file_width = int(frame.shape[1])
    file_height = int(frame.shape[0])
    frame = np.uint8(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    average_hsvValue = hsv[:,:,1].sum() / (file_width * file_height)
    averageSaturationCurrentFrame = (average_hsvValue*100)/256

    return float(averageSaturationCurrentFrame)


def calc_blur_features(frame, debug=False):
    """
    estimates blurriness using Laplacian filter,

    based on julian zebelein's master thesis
    """
    def variance_of_laplacian(image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F, ksize=5).var()

    file_width = int(frame.shape[1])
    file_height = int(frame.shape[0])
    frame = np.uint8(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prefinal = cv2.resize(gray, (file_width, file_height))
    #final = cv2.GaussianBlur(prefinal,(5,5),0)
    final = cv2.bilateralFilter(prefinal, 9, 75, 75)
    fm = variance_of_laplacian(final)
    return float(fm)


def calc_brisque_features(image, gray=False):
    """
    calcualte brisque no-reference features,

    References
    ----------
    - scikit-video
    - Mittal, A. K. Moorthy and A. C. Bovik, "No-Reference Image Quality Assessment in the Spatial Domain"
      In: IEEE Transactions on Image Processing, 2012.
    - Mittal, A. K. Moorthy and A. C. Bovik, "Referenceless Image Spatial Quality Evaluation Engine,"
      In: 45th Asilomar Conference on Signals, Systems and Computers , November 2011.
    """
    if not gray:
        image = skimage.color.rgb2gray(image)
    brisque_features = skvideo.measure.brisque_features(image)
    return brisque_features.tolist()[0]


def calc_niqe_features(image, gray=False):
    """
    calculate niqe features

    References
    ----------
    - scikit-video
    - Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik. "Making a ‘completely blind’ image quality analyzer."
      In: IEEE Signal Processing Letters 20.3 (2013): 209-212.
    """
    if not gray:
        image = skimage.color.rgb2gray(image)
    niqe = skvideo.measure.niqe(image)
    return float(niqe[0])


def ceiq(image, gray=False):
    """
    re-implementation and extension of https://github.com/mtobeiyf/CEIQ/blob/master/CEIQ.m
    features for "No-Reference Quality Assessment of Contrast-Distorted Images using Contrast Enhancement"

    References
    ----------
    - Jia Yan, Jie Li, Xin Fu: "No-Reference Quality Assessment of Contrast-Distorted Images using Contrast Enhancement",
      In: Journal of Visual Communication and Image Representation, 2018
    """
    if not gray:
        image = skimage.color.rgb2gray(image)
    # equalization
    image_eq = exposure.equalize_hist(image)

    # calculate similarity feature
    f1 = skimage.measure.compare_ssim(image, image_eq)

    # create histograms
    h1, _ = np.histogram(image.flatten(), bins=128)
    h2, _ = np.histogram(image_eq.flatten(), bins=128)

    # normalize histograms
    h1 = h1 / (image.shape[0] * image.shape[1])
    h2 = h2 / (image.shape[0] * image.shape[1])

    # select only positive values
    selection = (h1 > 0) & (h2 > 0)
    h1 = h1[selection]
    h2 = h2[selection]

    # calculate logs
    log_h1 = np.log2(h1)
    log_h2 = np.log2(h2)

    # these features are somehow entropy
    f2 = - sum(h1 *  log_h1)
    f3 = - sum(h2 *  log_h2)
    f4 = - sum(h1 *  log_h2)
    f5 = - sum(h2 *  log_h1)

    # additional features, that are not in the paper and added by stg7
    add1 = skimage.measure.compare_psnr(image, image_eq)
    hd = h1 - h2
    add2 = hd.max()
    add3 = hd.min()
    add4 = hd.mean()
    add5 = hd.std()

    return [f1, f2, f3, f4, f5, add1, add2, add3, add4, add5]