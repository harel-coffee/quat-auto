#!/usr/bin/env python3
"""
General video helper
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

import time

import skvideo.io
from skimage import img_as_uint
import scipy
import scipy.stats
import numpy as np
import pandas as pd
import numpy as np

from quat.log import *


def iterate_by_frame(video_filename, convert=True):
    """ iterator over all frames a video given by `video_filename`,
    if convert is true, than a conversion to uint will be performed

    Parameters
    ----------
    video_filename : str
        filename of a video file
    convert : bool
        if true performs a 8 bit conversion of each frame

    Returns
    -------
    interator for all video frames
    """
    for frame in skvideo.io.vreader(video_filename, verbosity=0):
        if convert:
            yield img_as_uint(frame)
        else:
            yield frame


def iterate_by_frame_two_videos(distortedvideo, referencevideo, convert=True):
    """
    interates over a pair of videos (distortedvideo, referencevideo) and returns pairs of frames (dis_frame, ref_frame),
    if convert is true, uint conversion will be performed,
    Important, if videos don't have the same number of frames it will stop after min(frames(dis), frames(ref)) frames.
    """
    dis_it = iterate_by_frame(distortedvideo, convert)
    ref_it = iterate_by_frame(referencevideo, convert)
    try:
        while True:
            x = next(dis_it)
            y = next(ref_it)
            yield (x, y)
    except Exception as e:
        pass
    return


def read_videos_frame_by_frame(
    distortedvideo,
    referencevideo,
    per_frame_function,
    per_frame_function_additional_params={},
    debug=False,
    stepsize=15,
):
    """read two videos frame by frame, and perform a function with parameters on each pair of frames
    """
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
                results.append(
                    per_frame_function(
                        x, y, last_x, last_y, **per_frame_function_additional_params
                    )
                )
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
    """ advanced_pooling temporal pooling method,
    """
    if len(x) > 0 and type(x[0]) in [dict, list]:
        res = {}
        df = pd.DataFrame(x)
        for k in df.columns:
            res = dict(
                res,
                **advanced_pooling(
                    df[k], name + "_" + str(k), parts=3, stats=stats, minimal=minimal
                ),
            )
        return res

    values = np.array(x)
    # filter only not nans and not inf values
    values = values[~np.isnan(values)]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        values = np.array([np.finfo(np.float32).max - 1])

    last_value = values[-1]
    first_value = values[0]
    _max = values.max()
    """
    _max_norm =  if values.max() != 0 else 1
    values = values / _max_norm
    """
    res = {
        f"{name}_mean": float(values.mean()),
        f"{name}_std": float(values.std()),
        f"{name}_first_value": float(first_value),
    }
    if not minimal:
        res = dict(
            res,
            **{
                f"{name}_last_value": float(last_value),
                f"{name}_max": float(_max),
                f"{name}_skew": float(scipy.stats.skew(values)),
                f"{name}_kurtosis": float(scipy.stats.kurtosis(values)),
                f"{name}_iqr": float(scipy.stats.iqr(values)),
            },
        )

    # split values in `parts` groups, and calculate mean, std
    groups = np.array_split(values, parts)
    for i in range(len(groups)):
        res[f"{name}_p{i}.mean"] = groups[i].mean()
        res[f"{name}_p{i}.std"] = groups[i].std()
    if stats:
        for i in range(11):
            quantile = round(0.1 * i, 1)
            res[f"{name}_{quantile}_quantil"] = float(
                np.percentile(values, 100 * quantile)
            )
    return res


def calc_per_second_scores(per_frame_scores, segment_duration):
    """ converts per frame scores to per second scores, using mean values per second
    """
    per_second_scores = []
    sec_groups = np.array_split(per_frame_scores, np.ceil(segment_duration))
    for sec in range(len(sec_groups)):
        per_second_scores.append(float(sec_groups[sec].mean()))

    return per_second_scores
