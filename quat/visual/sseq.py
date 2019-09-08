#!/usr/bin/env python3
"""
Re-implementation of sseq features based on matlab version and the research paper, see

References
----------
- Lixiong Liu, Bao Liu, Hua Huang, and Alan Conrad Bovik, "No-reference Image Quality Assessment Based on Spatial and Spectral Entropies",
  In: Signal Processing: Image Communication, Vol. 29, No. 8, pp. 856-863, Sep. 2014.
- Lixiong Liu, Bao Liu, Hua Huang, and Alan Conrad Bovik, "SSEQ Software Release",
  URL: http://live.ece.utexas.edu/research/quality/SSEQ_release.zip, 2014

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

import os
import sys
import argparse
import time
import math
import multiprocessing
from multiprocessing import Pool

import numpy as np
import pandas as pd

import skimage.color
import skimage.io
import scipy.stats
import skimage.transform
import skimage.measure
import skimage.filters.rank
import skimage.morphology
import skimage.util.shape

try:
    from ..log import *
except:
    import sys
    sys.path.append(os.path.dirname(__file__) + '/../')
    from log import *


def per_block(image, function, block_shape=(8, 8)):
    """ truncate image size to multiple of block_shape,
    otherwise view_as_blocks will not work
    """
    image = image[:image.shape[0] - image.shape[0] % block_shape[0],
                  :image.shape[1] - image.shape[1] % block_shape[1]]
    rows = []
    for row in skimage.util.shape.view_as_blocks(image, block_shape):
        cols = []
        for col in row:
            cols.append(function(col))
        rows.append(cols)
    return np.array(rows)


def calc_sseq_features(image, scale=3, gray=False):
    """ calculate sseq features of an images with several scales """
    def calc_stats(x, weight=[0.2, 0.8]):
        x = x.flatten()
        x.sort()
        t = x[math.ceil(len(x) * weight[0]): math.ceil(len(x) * weight[1])]
        mu = t.mean()
        ske = scipy.stats.skew(t)
        return [mu, ske]

    def spectral(x):
        rf = scipy.fftpack.dct(x, type=2)
        rf[0, 0] = 0.00000001
        nrf = np.square(rf) / np.square(rf).sum()
        nrf[nrf == 0] = 0.00000001
        return skimage.measure.shannon_entropy(nrf)

    features = []
    if not gray:
        img_gray = skimage.color.rgb2gray(image)
    else:
        img_gray = image

    for i in range(0, scale):
        spatial_entropy = per_block(img_gray, skimage.measure.shannon_entropy, (8, 8))
        features.extend(calc_stats(spatial_entropy))

        spectral_entropy = per_block(img_gray, spectral, (8, 8))
        features.extend(calc_stats(spectral_entropy))
        img_gray = skimage.transform.rescale(img_gray, 0.5, mode="reflect")

    res = {}
    for i, x in enumerate(features):
        res["sseq_" + str(i)] = x
    return res


def sseq_features(image_filename, scale=3):
    """ sseq features of an image file """
    img = skimage.io.imread(image_filename)
    res = calc_sseq_features(img, scale)
    return dict({"filename": image_filename}, **res)


def main(_):
    """ extract image features """
    # argument parsing
    parser = argparse.ArgumentParser(description='calculate sseq no reference metric features',
                                     epilog="stg7 2018",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputimg", type=str, nargs="+", help="input img for training")
    parser.add_argument('--output_file', type=str, default="sseq_features.csv", help="output_file for storing calculated features")
    parser.add_argument('--cpu_count', type=int, default=multiprocessing.cpu_count(), help="cpus/threads that are used for processing")

    argsdict = vars(parser.parse_args())

    start_time = time.time()

    cpu_count = argsdict["cpu_count"]
    pool = Pool(processes=cpu_count)
    lInfo("calculate no reference metric features for {} images".format(len(argsdict["inputimg"])))

    params = [image_filename for image_filename in argsdict["inputimg"]]
    results = pool.map(sseq_features, params)

    df = pd.DataFrame(results)

    df.to_csv(argsdict["output_file"], index=False)
    lInfo("feature extraction done: {} s".format(time.time() - start_time))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
