#!/usr/bin/env python3
import skvideo
import skvideo.measure

import skimage.color
import skimage.io
import skimage.transform
from skimage import img_as_ubyte

from .base_features import Feature

class SSIM(Feature):
    def calc_ref_dis(self, dis, ref):
        x_g = skimage.color.rgb2gray(ref)
        y_g = skimage.color.rgb2gray(dis)
        v = float(skvideo.measure.ssim(x_g, y_g, bitdepth=10, scaleFix=False))
        self._values.append(v)
        return v

    def fullref(self):
        return True


class PSNR(Feature):
    def calc_ref_dis(self, dis, ref):
        x_g = skimage.color.rgb2gray(ref)
        y_g = skimage.color.rgb2gray(dis)
        v = float(skvideo.measure.psnr(x_g, y_g, bitdepth=10))
        self._values.append(v)
        return v

    def fullref(self):
        return True


class VIFP(Feature):
    def calc_ref_dis(self, dis, ref):
        v = vifp_mscale(ref, dis, 4)
        self._values.append(v)
        return v

    def fullref(self):
        return True