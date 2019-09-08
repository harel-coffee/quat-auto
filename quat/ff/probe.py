#!/usr/bin/env python3
"""
ffprobe related helpers
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
import shutil
import os
import json

from quat.utils.system import shell_call


def ffprobe(filename):
    """ run ffprobe to get some information of a given video file
    """
    if shutil.which("ffprobe") is None:
        raise Exception("you need to have ffprobe installed, please read README.md.")

    if not os.path.isfile(filename):
        raise Exception("{} is not a valid file".format(filename))

    cmd = "ffprobe -show_format -select_streams v:0 -show_streams -of json '{filename}' 2>/dev/null".format(filename=filename)

    res = shell_call(cmd).strip()

    if len(res) == 0:
        raise Exception("{} is somehow not valid, so ffprobe could not extract anything".format(filename))

    res = json.loads(res)

    needed = {"pix_fmt": "unknown",
              "bits_per_raw_sample": "unknown",
              "width": "unknown",
              "height": "unknown",
              "avg_frame_rate": "unknown",
              "codec_name": "unknown"
             }
    for stream in res["streams"]:
        for n in needed:
            if n in stream:
                needed[n] = stream[n]
                if n == "avg_frame_rate":  # convert framerate to numeric integer value
                    needed[n] = round(eval(needed[n]))
    needed["bitrate"] = res.get("format", {}).get("bit_rate", -1)
    needed["codec"] = needed["codec_name"]

    return needed


def ffprobe_framesizes_types(filename):
    """ run ffprobe to get some frame information of a given video file
    """
    if shutil.which("ffprobe") is None:
        raise Exception("you need to have ffprobe installed, please read README.md.")

    if not os.path.isfile(filename):
        raise Exception("{} is not a valid file".format(filename))

    cmd = "ffprobe -show_format -select_streams v:0 -show_frames -show_entries frame=pkt_pts_time,pkt_dts_time,pkt_duration_time,pkt_size,pict_type -of json '{filename}' 2>/dev/null".format(filename=filename)
    res = shell_call(cmd).strip()

    if len(res) == 0:
        raise Exception("{} is somehow not valid, so ffprobe could not extract anything".format(filename))

    res = json.loads(res)
    return res



def run_ffprobe(video_file):
    cmd = """ffprobe -loglevel error -select_streams v -show_frames -show_entries frame=pkt_pts_time,pkt_dts_time,pkt_duration_time,pkt_size,pict_type -of json {}"""
    cmd = cmd.format(video_file)

    res = json.loads(shell_call(cmd))
    res["file"] = video_file
    return res