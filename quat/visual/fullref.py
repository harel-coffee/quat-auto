#!/usr/bin/env python3
"""
Full reference features.

All full-reference features share a common interface,
e.g. in case of SSIM,

.. code-block:: python

    ...
    ssim = SSIM()
    v = ssim.calc_ref_dis(dis_frame, ref_frame)
    # will return the calculated ssim value in `v`

TODO: add references to metrics
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
import numpy as np
import skvideo
import skvideo.measure

import skimage.color
import skimage.io
import skimage.transform
from skimage import img_as_ubyte

from .base_features import Feature
from .vifp import vifp_mscale


class SSIM(Feature):
    """ Calculate SSIM """

    def calc_ref_dis(self, dis, ref):
        """ calculates ssim score """
        x_g = skimage.color.rgb2gray(ref)
        y_g = skimage.color.rgb2gray(dis)
        v = float(skvideo.measure.ssim(x_g, y_g, bitdepth=10, scaleFix=False))
        self._values.append(v)
        return v

    def fullref(self):
        return True


class PSNR(Feature):
    """ Caclulate PSNR """

    def psnr(self, referenceVideoData, distortedVideoData, bitdepth=8):
        """ a minimal adjusted variant of psnr from scikit video """
        bitdepth = np.int64(bitdepth)
        maxvalue = np.int64(2**bitdepth - 1)
        maxsq = maxvalue**2

        referenceFrame = referenceVideoData.astype(np.float64)
        distortedFrame = distortedVideoData.astype(np.float64)

        mse = np.mean((referenceFrame - distortedFrame)**2)
        psnr = 10 * np.log10(maxsq / mse)
        return psnr

    def calc_ref_dis(self, dis, ref):
        """ calculates psnr score """
        x_g = skimage.color.rgb2gray(ref)
        y_g = skimage.color.rgb2gray(dis)
        v = float(self.psnr(x_g, y_g, bitdepth=10))
        self._values.append(v)
        return v

    def fullref(self):
        return True


class VIFP(Feature):
    """ Caclulate multi scale (4 scales) VIFP """

    def calc_ref_dis(self, dis, ref):
        """ calculates ms-vifp scores """
        v = vifp_mscale(ref, dis, 4)
        self._values.append(v)
        return v

    def fullref(self):
        return True


class ResolutionSimilarities(Feature):
    """ Tries to estimate resolution of the distorted video
    """

    def calc_ref_dis(self, dis, ref):
        """ calculates resoltion based on similarity measurement """
        x_g = skimage.color.rgb2gray(ref).astype(np.float32)
        y_g = skimage.color.rgb2gray(dis).astype(np.float32)
        resolutions = [2160, 1440, 1080, 720, 480, 360, 240, 144]
        # resolutions = list(range(2160, 140, -32))
        scale_factors = [x / resolutions[0] for x in resolutions]
        # print("scale_factors", scale_factors)
        height = max(x_g.shape[0], y_g.shape[0])
        width = max(x_g.shape[1], y_g.shape[1])
        # print("height", height)
        aspect = width / height
        values = []
        for sc in scale_factors:
            r = round(sc * height)
            x_gs = skimage.transform.resize(x_g, (r, round(aspect * r)), mode="reflect")
            x_gs = skimage.transform.resize(
                x_gs, (height, round(aspect * height)), mode="reflect"
            )
            v = float(skvideo.measure.mse(x_gs, y_g)[0])  # ** 2 / sc
            values.append(v)
        values = np.array(values)
        res = resolutions[np.argmin(values)]
        self._values.append(res)
        # print(res)
        return res

    def fullref(self):
        return True


class FramerateEstimator(Feature):
    """
    based on frame differences of src and distorted video estimate framerate of distorted video

    TODO: check: could be also an no-reference feature
    """

    WINDOW = 60  # maximum number of frames in sliding window

    def __init__(self):
        self._calculated_values = []
        self._values = []
        self._lastframe_ref = None
        self._lastframe_dis = None

    def rmse(self, x, y):
        return np.sqrt(((x - y) ** 2).mean())

    def calc_ref_dis(self, dis, ref):
        """ tries to esimate fps"""
        v = {"ref": 0, "dis": 0}
        if self._lastframe_ref is not None and self._lastframe_dis is not None:
            v = {
                "ref": self.rmse(self._lastframe_ref.flatten(), ref.flatten()),
                "dis": self.rmse(self._lastframe_dis.flatten(), dis.flatten()),
            }
        self._lastframe_ref = ref.copy()
        self._lastframe_dis = dis.copy()
        self._calculated_values.append(v)
        if len(self._calculated_values) > self.WINDOW:
            self._calculated_values = self._calculated_values[1:]

        zeros_ref = sum(np.array([x["ref"] for x in self._calculated_values]) == 0)
        zeros_dis = sum(np.array([x["dis"] for x in self._calculated_values]) == 0)

        fps = int(len(self._calculated_values) - zeros_dis + zeros_ref)

        self._values.append(fps)
        return fps

    def fullref(self):
        return True

    def get_values(self):
        return self._values[self.WINDOW :]
