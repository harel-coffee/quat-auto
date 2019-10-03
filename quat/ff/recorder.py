#!/usr/bin/env python3
"""
FFmpeg based screen recorder

# TODO: audio is ususally not synced
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
import os
import time
import subprocess
import signal

from quat.log import *


class Recorder:
    """
    ffmpeg based screen recording
    """

    def __init__(self, resultfolder="tmp"):
        """
        creates a ffmpeg screen recorder

        Parameters
        ----------
        resultfolder : str
            folder to store the resulting record
        """
        os.makedirs(resultfolder, exist_ok=True)
        self._resultfolder = resultfolder

    def _create_record_command(
        self, width, height, shift, outputfile, fps, hardwareAcceleration=True
    ):
        # no hardware acceleration
        cmd = f"""
        ffmpeg
            -loglevel quiet
            -hide_banner
            -draw_mouse 0
            -y
            -video_size {width}x{height}
            -framerate {fps}
            -rtbufsize 2G
            -probesize 1G
            -f x11grab
            -thread_queue_size 512
            -i :0.0{shift}
            -f pulse
            -ac 2
            -thread_queue_size 512
            -i 0
            -c:v libx264
            -crf 22
            -preset ultrafast
            -pix_fmt yuv420p
            -c:a aac
            -q:a 2
            '{outputfile}'
            """
        if hardwareAcceleration:
            cmd = f"""
            ffmpeg
                -loglevel quiet
                -hide_banner
                -draw_mouse 0
                -y
                -video_size {width}x{height}
                -framerate {fps}
                -rtbufsize 2G
                -probesize 1G
                -vaapi_device /dev/dri/renderD129
                -f x11grab
                -thread_queue_size 512
                -i :0.0{shift}
                -f pulse
                -ac 2
                -thread_queue_size 512
                -i 0
                -vf 'format=nv12,hwupload'
                -c:v h264_vaapi
                -qp 22
                -pix_fmt vaapi_vld
                -c:a aac
                -q:a 2
                '{outputfile}'
                """
        cmd = " ".join(cmd.split())
        lDbg(cmd)
        return cmd

    def start(
        self,
        filebasename,
        hardwareAcceleration=False,
        width=1366,
        height=768,
        fps=24,
        shift="",
    ):
        """
        starts the screen record

        Parameters
        ----------
        filebasename : str
            basename of recording filename (will be extended by .mkv and stored in the recordfolder)
        hardwareAcceleration : bool
            if true try to used hardware acceleration for intel graphics cards (TODO: experimental)
        width : int
            recording width (screen resolution must be at least having this width)
        height : int
            recording height
        fps : int
            frames per second to record
        shift : str
            ffmpeg specific shift, e.g. "+100,200" TODO: explain better
        """
        outputfile = os.path.join(self._resultfolder, filebasename + ".mkv")

        cmd = self._create_record_command(
            width, height, shift, outputfile, fps, hardwareAcceleration
        )

        lInfo("start recording")
        self._process = subprocess.Popen(
            cmd,
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    def stop(self):
        """
        stop recording
        """
        lInfo("stop recording")
        os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
        lDbg(self._process.communicate())


if __name__ == "__main__":
    """ example usage of recoder """
    lInfo("internal test for recorder")
    recorder = Recorder(resultfolder="recordings")
    recorder.start("testfile")
    i = 0
    try:
        while True:
            lInfo(i)
            time.sleep(1)
            i += 1

    except KeyboardInterrupt:
        lInfo("stop recording")
    finally:
        recorder.stop()
