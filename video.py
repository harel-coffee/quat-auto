#!/usr/bin/env python3
import time

import skvideo.io
from skimage import img_as_uint
import scipy
import scipy.stats
import numpy as np
import pandas as pd
import numpy as np

from .log import *


def iterate_by_frame(video_filename, convert=True):
    """ iterator over all frames a video_filename """
    for frame in skvideo.io.vreader(video_filename, verbosity=0):
        if convert:
            yield img_as_uint(frame)
        else:
            yield frame


def iterate_by_frame_two_videos(distortedvideo, referencevideo, convert=True):
    dis_it = iterate_by_frame(distortedvideo, convert)
    ref_it = iterate_by_frame(referencevideo, convert)
    try:
        while True:
            x = next(dis_it)
            y = next(ref_it)
            yield (x, y)
    except Exception as e:
        pass
    raise StopIteration


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


def advanced_pooling(x, name, parts=3, stats=True, minimal=False):
    if len(x) > 0 and type(x[0]) == dict:
        res = {}
        for k in x[0]:
            df = pd.DataFrame(x)
            res = dict(res, **advanced_pooling(df[k], name + "_" + k, parts=3, stats=stats, minimal=minimal))
        return res

    values = np.array(x)
    # filter only not nans and not inf values
    values = values[~np.isnan(values)]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        values = np.array([np.finfo(np.float32).max - 1])

    last_value = values[-1]
    first_value = values[-1]
    _max = values.max() if values.max() != 0 else 1
    """
    values = values / _max
    """
    res = {
        f"{name}_mean": float(values.mean()),
        f"{name}_std": float(values.std()),
        f"{name}_first_value": float(first_value),
    }
    if not minimal:
        res = dict(res, **{
            f"{name}_last_value": float(last_value),
            f"{name}_max": float(_max),
            f"{name}_skew": float(scipy.stats.skew(values)),
            f"{name}_kurtosis": float(scipy.stats.kurtosis(values)),
            f"{name}_iqr": float(scipy.stats.iqr(values)),
        })

    # split values in `parts` groups, and calculate mean, std
    groups = np.array_split(values, parts)
    for i in range(len(groups)):
        res[f"{name}_p{i}.mean"] = groups[i].mean()
        res[f"{name}_p{i}.std"] = groups[i].std()
    if stats:
        for i in range(11):
            quantile = round(0.1 * i, 1)
            res[f"{name}_{quantile}_quantil"] = float(np.percentile(values, 100 * quantile))
    return res


def calc_per_second_scores(per_frame_scores, segment_duration):
    """ converts per frame scores to per second scores
    """
    per_second_scores = []
    sec_groups = np.array_split(per_frame_scores, np.ceil(segment_duration))
    for sec in range(len(sec_groups)):
        per_second_scores.append(float(sec_groups[sec].mean()))

    return per_second_scores

