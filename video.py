#!/usr/bin/env python3
import time

import skvideo.io
from skimage import img_as_uint

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

