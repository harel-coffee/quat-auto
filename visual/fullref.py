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


class ResolutionSimilarities(Feature):
    """ try to estimate resolution of the distorted video
    """
    def calc_ref_dis(self, dis, ref):
        x_g = skimage.color.rgb2gray(ref).astype(np.float32)
        y_g = skimage.color.rgb2gray(dis).astype(np.float32)
        resolutions = [2160, 1440, 1080, 720, 480, 360, 240, 144]
        #resolutions = list(range(2160, 140, -32))
        scale_factors = [x / resolutions[0] for x in resolutions]
        #print("scale_factors", scale_factors)
        height = max(x_g.shape[0], y_g.shape[0])
        width = max(x_g.shape[1], y_g.shape[1])
        #print("height", height)
        aspect = width / height
        values = []
        for sc in scale_factors:
            r = round(sc * height)
            x_gs = skimage.transform.resize(x_g, (r, round(aspect * r)), mode='reflect')
            x_gs = skimage.transform.resize(x_gs, (height, round(aspect * height)), mode='reflect')
            v = float(skvideo.measure.mse(x_gs, y_g)[0]) #** 2 / sc
            values.append(v)
        values = np.array(values)
        res = resolutions[np.argmin(values)]
        self._values.append(res)
        return res

    def fullref(self):
        return True