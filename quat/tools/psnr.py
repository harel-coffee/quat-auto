#!/usr/bin/env python3
"""
measure psnr
TODO/FIXME: use quat methods
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

import sys
import os
import argparse
import json
import glob
from multiprocessing import Pool
import multiprocessing
import time
import bz2
import gzip

import numpy as np
import skvideo
import skvideo.io
import skimage.transform
import skimage.io


def file_open(filename, mode="r"):
    """ Open a file, and if you add bz2 to filename a compressed file will be opened
    """
    if "bz2" in filename:
        return bz2.open(filename, mode + "t")
    if "gz" in filename:
        return gzip.open(filename, mode + "t")
    return open(filename, mode)


def iterate_by_frame(video_filename):
    """ iterator over all frames a video_filename """
    for frame in skvideo.io.vreader(video_filename):
        yield frame
    raise StopIteration


def colorgreen(m):
    return "\033[92m" + m + "\033[0m"


def lInfo(msg):
    print(colorgreen("[INFO ] ") + str(msg), flush=True)


def read_videos_frame_by_frame(distortedvideo, referencevideo, per_frame_function, per_frame_function_additional_params={}, debug=False):
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
            results.append(per_frame_function(x, y, last_x, last_y, **per_frame_function_additional_params))
            k += 1
            last_x = x
            last_y = y
            if k == 50 and debug:
                break
            time_per_frame = time.time() - start_time
            lInfo("time per frame: {}".format(time_per_frame))

    except StopIteration:
        lInfo("reached end")
    return results


def calc_psnr(x, y, last_x, last_y, bitdepth = 10):
    new_shape = (max(x.shape[0], y.shape[0]), max(x.shape[1], y.shape[1]), max(x.shape[2], y.shape[2]))

    if new_shape != x.shape:
        x = skimage.transform.resize(x, new_shape[0:2], mode='reflect')
    if new_shape != y.shape:
        y = skimage.transform.resize(y, new_shape[0:2], mode='reflect')

    maxvalue = np.int(2 ** bitdepth - 1)
    maxsq = maxvalue ** 2

    mse = np.mean((x - y) ** 2)
    psnr = 10 * np.log10(maxsq / mse)
    lInfo("psnr: {}".format(psnr))
    return psnr


def psnr_report(video, reference, output_dir):
    reportname = output_dir + "/" + os.path.splitext(os.path.basename(video))[0] + "_psnr.json.bz2"
    results = read_videos_frame_by_frame(video, reference, calc_psnr)
    values = {}
    values["per_frame"] = results
    r = np.array(results)
    values["mean"] = r.mean()
    values["median"] = np.median(r)
    values["max"] = r.max()
    values["min"] = r.min()
    values["std"] = r.std()

    with file_open(reportname, "w") as rep:
        json.dump(values, rep)

    return {"video": video, "psnr_mean": values["mean"]}


def main(_):
    parser = argparse.ArgumentParser(description='calculate psnr for videos with different resolutions',
                                     epilog="stg7 2017",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', type=str, nargs="+", help='video to measure')
    parser.add_argument('--referencevideo', type=str, default=None, help="reference video; required")
    parser.add_argument('--output_dir', type=str, default="psnr", help='output directory')
    parser.add_argument('--cpu_count', type=int, default=multiprocessing.cpu_count(), help='thread/cpu count')

    argsdict = vars(parser.parse_args())
    if argsdict["referencevideo"] is None:
        print("a referencevideo is required, see -h")
        return -1

    os.makedirs(argsdict["output_dir"], exist_ok=True)

    pool = Pool(argsdict["cpu_count"])
    params = [(video, argsdict["referencevideo"], argsdict["output_dir"]) for video in argsdict["video"]]
    res = pool.starmap(psnr_report, params)
    print(json.dumps(res, indent=4))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
