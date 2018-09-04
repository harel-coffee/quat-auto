#!/usr/bin/env python3
import time

import skvideo.io
from skimage import img_as_uint
import scipy
import scipy.stats
import numpy as np
import pandas as pd

from .log import *


def iterate_by_frame(video_filename, convert=True):
    """ iterator over all frames a video_filename """
    for frame in skvideo.io.vreader(video_filename, verbosity=0):
        if convert:
            yield img_as_uint(frame)
        else:
            yield frame


def read_videos_frame_by_frame(distortedvideo, referencevideo, per_frame_function, per_frame_function_additional_params={}, debug=False, stepsize=15):
    lInfo("handle : {} and {}".format(distortedvideo, referencevideo))
    distortedvideo_it = iterate_by_frame(distortedvideo)
    referencevideo_it = iterate_by_frame(referencevideo)
    k = 0
    results = []
    last_x = None
    last_y = None
    try:
        while True:
            lInfo("frame: {}".format(k))
            start_time = time.time()
            x = next(referencevideo_it)
            y = next(distortedvideo_it)
            per_frame_function_additional_params["distortedvideo"] = distortedvideo
            per_frame_function_additional_params["referencevideo"] = referencevideo

            if k % stepsize == 0:
                results.append(per_frame_function(x, y, last_x, last_y, **per_frame_function_additional_params))
            k += 1
            last_x = x
            last_y = y
            time_per_frame = time.time() - start_time
            lInfo("time per frame: {}".format(time_per_frame))
            if debug and k == 2:
                break

    except StopIteration:
        lInfo("reached end")
    return results


def advanced_pooling(x, name, parts=3):
    if len(x) > 0 and type(x[0]) == dict:
        res = {}
        for k in x[0]:
            df = pd.DataFrame(x)
            res = dict(res, **advanced_pooling(df[k], name, parts=3))
        return res

    values = np.array(x)
    values = values / values.max()
    res = {
        f"{name}_mean": float(values.mean()),
        f"{name}_std": float(values.std()),
        f"{name}_skew": float(scipy.stats.skew(values)),
        f"{name}_kurtosis": float(scipy.stats.kurtosis(values)),
        f"{name}_iqr": float(scipy.stats.iqr(values)),
    }

    # split values in `parts` groups, and calculate mean, std
    groups = np.array_split(values, parts)
    for i in range(len(groups)):
        res[f"{name}_p{i}.mean"] = groups[i].mean()
        res[f"{name}_p{i}.std"] = groups[i].std()

    for i in range(11):
        quantile = round(0.1 * i, 1)
        res[f"{name}_{quantile}_quantil"] = float(np.percentile(values, 100 * quantile))
    return res

