#!/usr/bin/env python3
"""
tool to extract brisque niqe feature values of a given video

Note: the extracted values are [0,1] normalized, due
to a float conversion of the given video frames,
if you need values in range [0,255], just multiply
the extracted values by 255
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
import multiprocessing

from quat.log import *
from quat.video import iterate_by_frame
from quat.parallel import run_parallel

from quat.visual.image import calc_brisque_features, calc_niqe_features
from quat.visual.base_features import ImageFeature
from quat.utils.fileutils import get_filename_without_extension
from quat.utils.fileutils import write_json
from quat.video import advanced_pooling

def extract_brisque_niqe(video):
    """
    extracts brisque niqe feature values of a given video


    Parameters
    ----------
    video : str
        filename of the video that should be processed

    Returns
    -------
    dict of per frame results, filename and temporally pooled feature values

    Examples
    --------
    >>> import quat.tools.siti extract_brisque_niqe
    >>> res = extract_brisque_niqe("myvideo.mkv")
    """
    features = {
        "brisque": ImageFeature(calc_brisque_features),
        "niqe": ImageFeature(calc_niqe_features)
    }
    results = []
    frame_number = 1
    for frame in iterate_by_frame(video, convert=True):
        r = {}
        for f in features:
            v = features[f].calc(frame)
            r[f] = v
        r["frame"] = frame_number
        results.append(r)
        jprint(r)
        frame_number += 1

    pooled = {}
    for f in features:
        pooled = dict(pooled, **advanced_pooling(features[f].get_values(), f))

    return {
        "video": video,
        "values": results,
        "pooled": pooled
    }


def main(_=[]):
    parser = argparse.ArgumentParser(
        description="calculate brisque niqe feature values for a given video",
        epilog="stg7 2019",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=str, nargs="+", help="video to measure")
    parser.add_argument(
        "--reportfolder",
        type=str,
        default="reports",
        help="folder to store all calculations",
    )
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=multiprocessing.cpu_count(),
        help="thread/cpu count",
    )

    a = vars(parser.parse_args())
    lInfo(f"cli arguments: {a}")

    res = run_parallel(
        a["video"],  # list of videofilenames
        extract_brisque_niqe,  # apply this function to each videofilename
        num_cpus=a["cpu_count"],
    )

    os.makedirs(a["reportfolder"], exist_ok=True)
    for r in res:
        basename = get_filename_without_extension(r["video"])
        reportname = os.path.join(a["reportfolder"], basename + ".json")
        write_json(r["values"], reportname, prettify=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
