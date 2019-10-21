#!/usr/bin/env python3
"""
ffmpeg methods to convert, rescale, and center crop a given video
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
import sys
import argparse
import glob
import shutil

from quat.log import *


def check_ffmpeg():
    """
    checks if ffmpeg is installed in the system

    Returns
    -------
    true if ffmpeg can be used, otherwise an Exception is thrown
    """
    if shutil.which("ffmpeg") is None:
        raise Exception("ffmpeg not found")
    return True


def __run_multi_line_cmd(cmd):
    """
    run a command that consists of several lines that are combined again

    Parameters
    ----------
    cmd : str
        command to run, e.g. cmd="ls \n -la" will run "ls -la"

    TODO: move to utils/system?

    Returns
    -------
    in case of error an error is thrown
    """
    # remove multiple spaces in cmd
    cmd = " ".join(cmd.split())
    ret = os.system(cmd)
    if ret != 0:
        raise Exception(f"error in running command {cmd}")


def crop_video(input_file, tmp_folder, ccheight=360):
    """
    create a center cropped version of a video

    Parameters
    ----------
    input_file : str
        input video file
    tmp_folder : str
        folder where center cropped version is store, this version gets "_cropped.mkv" as suffix
    ccheight : int
        default=360, height of the center crop

    Returns
    -------
    filename and path of generated center cropped video
    """
    check_ffmpeg()
    os.makedirs(tmp_folder, exist_ok=True)

    output_file = os.path.join(
        tmp_folder, os.path.splitext(os.path.basename(input_file))[0] + "_cropped.mkv"
    )
    cmd = f"""
    ffmpeg -nostdin -loglevel quiet
    -y
    -i {input_file}
    -filter:v crop=ccheight*in_w/in_h:ccheight
    -c:v ffvhuff -an
    {output_file} 2>/dev/null"""

    lInfo(f"crop video: {input_file} to {output_file}")
    __run_multi_line_cmd(cmd)
    return output_file


def rescale_video(input_file, tmp_folder, height=360):
    """
    rescales a given video
    """
    check_ffmpeg()
    os.makedirs(tmp_folder, exist_ok=True)

    output_file = os.path.join(
        tmp_folder, os.path.splitext(os.path.basename(input_file))[0] + "_rescaled.mkv"
    )
    cmd = f"""
    ffmpeg -nostdin -loglevel quiet
    -y
    -i {input_file}
    -filter:v scale=-2:{height}
    -c:v ffvhuff -an
    {output_file}  2>/dev/null"""

    lInfo(f"rescale video: {input_file} to {output_file}")
    __run_multi_line_cmd(cmd)
    return output_file


def convert_to_avpvs(
    input_file, tmp_folder, framerate="60/1", width=3840, height=-2, pix_fmt="yuv420p"
):
    """
    converts a video to a unified resolution, framerate and pixel format,
    can be used, e.g. in case of a full reference model, to unify a distorted video with the source video

    Parameters
    ----------
    input_file : str
        input video file
    tmp_folder : str
        folder where converted video is stored
    framerate : str
        framerate of final video
    width : int
        width of final video
    height : int
        height of final video, use -2 to automatically determine height based on width
    pix_fmt : str
        pixel format of final video

    Returns
    -------
    filename and path of the converted video
    """
    check_ffmpeg()
    os.makedirs(tmp_folder, exist_ok=True)

    output_file = os.path.join(
        tmp_folder, os.path.splitext(os.path.basename(input_file))[0] + ".mkv"
    )
    cmd = f"""
    ffmpeg -nostdin -loglevel quiet
    -y
    -i {input_file}
    -filter:v scale={width}:{height},fps={framerate},setsar=1/1
    -c:v ffvhuff
    -an
    -pix_fmt {target_pix_fmt}
    {output_file} 2>/dev/null"""

    lInfo(f"convert to avpvs: {input_file} to {output_file}")
    __run_multi_line_cmd(cmd)
    return output_file


def convert_to_avpvs_and_crop(
    input_file,
    tmp_folder,
    framerate="60/1",
    width=3840,
    height=-2,
    pix_fmt="yuv420p",
    ccheight=360,
):
    """
    converts a video to a unified resolution, framerate and pixel format and
    performs afterwards a center cropping
    can be used, e.g. in case of a full reference model, to unify a distorted video with the source video

    Parameters
    ----------
    input_file : str
        input video file
    tmp_folder : str
        folder where converted video is stored
    framerate : str
        framerate of final video
    width : int
        width of final video
    height : int
        height of final video, use -2 to automatically determine height based on width
    pix_fmt : str
        pixel format of final video
    ccheight : int
        center crop height of final crop
    Returns
    -------
    filename and path of the converted and center cropped video
    """
    check_ffmpeg()
    lInfo(
        f"avpvs + cropping generation with: {width}x{height}@{framerate}-{pix_fmt} using ccheight:{ccheight}"
    )

    os.makedirs(tmp_folder, exist_ok=True)

    output_file = os.path.join(
        tmp_folder, os.path.splitext(os.path.basename(input_file))[0] + ".mkv"
    )

    cmd = f"""
    ffmpeg -nostdin -loglevel quiet -threads 4
    -y
    -i {input_file}
    -filter:v scale={width}:{height},fps={framerate},setsar=1/1
    -an
    -pix_fmt {pix_fmt}
    -f yuv4mpegpipe pipe:
    |
    ffmpeg -nostdin -loglevel quiet -threads 4
    -y -f yuv4mpegpipe
    -i pipe:
    -filter:v crop={ccheight}*in_w/in_h:{ccheight}
    -c:v ffvhuff
    -an
    {output_file} 2>/dev/null"""

    lInfo(f"convert to cropped-avpvs: {input_file} to {output_file}")

    __run_multi_line_cmd(cmd)
    return output_file
