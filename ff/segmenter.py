#!/usr/bin/env python3
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
    return sorted(lglob(f"{output_folder}/{basename}_*.mp4"))