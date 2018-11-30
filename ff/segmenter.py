#!/usr/bin/env python3
import shutil
import os
import glob

from ..utils.system import shell_call
from ..utils.system import lglob



def create_segments(videofilename, output_folder, segment_time=4, debug=False):
    """
    segment a video
    """
    basename = os.path.basename(videofilename)
    os.makedirs(output_folder, exist_ok=True)
    cmd = f"""ffmpeg -nostdin -loglevel quiet -threads 4 -y -i {videofilename} -c:v copy -c:a copy -segment_time {segment_time} -f segment {output_folder}/{basename}_%08d.mp4 2>/dev/null"""
    if debug:
        print(cmd)
    shell_call(cmd)
    return lglob(f"{output_folder}/{basename}_*.mp4")