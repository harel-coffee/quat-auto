#!/usr/bin/env python3
"""
measure psnr
TODO/FIXME: use quat methods, maybe remove
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

from quat.log import *
from quat.video import iterate_by_frame_two_videos

from quat.utils.fileutils import file_open
from quat.visual.fullref import PSNR
from quat.utils.fileutils import get_filename_without_extension
from quat.utils.fileutils import write_json
from quat.video import advanced_pooling


def psnr_report(video, reference, output_dir):
    reportname = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(video))[0] + "_psnr.json.bz2"
    )

    features = {
        "psnr": PSNR(),
    }

    results = []
    frame_number = 1
    for dis, ref in iterate_by_frame_two_videos(video, reference, convert=True):
        # upscale frames to same size, if needed
        new_shape = (
            max(dis.shape[0], ref.shape[0]),
            max(dis.shape[1], ref.shape[1]),
            max(dis.shape[2], ref.shape[2]),
        )

        if new_shape != dis.shape:
            dis = skimage.transform.resize(dis, new_shape[0:2], mode="reflect")
        if new_shape != ref.shape:
            ref = skimage.transform.resize(ref, new_shape[0:2], mode="reflect")

        # calculate scores
        r = {}
        for f in features:
            v = features[f].calc_ref_dis(dis, ref)
            r[f] = v

        r["frame"] = frame_number
        results.append(r)
        jprint(r)
        frame_number += 1

    pooled = {}
    for f in features:
        pooled = dict(pooled, **advanced_pooling(features[f].get_values(), f))

    res = {
        "video": video,
        "values": results,
        "pooled": pooled
    }

    with file_open(reportname, "w") as rep:
        json.dump(res, rep)

    return res


def main(_=[]):
    parser = argparse.ArgumentParser(
        description="calculate psnr for videos with different resolutions",
        epilog="stg7 2017",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=str, nargs="+", help="video to measure")
    parser.add_argument(
        "--referencevideo", type=str, default=None, help="reference video; required"
    )
    parser.add_argument(
        "--output_dir", type=str, default="psnr", help="output directory"
    )
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=multiprocessing.cpu_count(),
        help="thread/cpu count",
    )

    argsdict = vars(parser.parse_args())
    if argsdict["referencevideo"] is None:
        print("a referencevideo is required, see -h")
        return -1

    os.makedirs(argsdict["output_dir"], exist_ok=True)

    pool = Pool(argsdict["cpu_count"])
    params = [
        (video, argsdict["referencevideo"], argsdict["output_dir"])
        for video in argsdict["video"]
    ]
    res = pool.starmap(psnr_report, params)
    print(json.dumps(res, indent=4))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
