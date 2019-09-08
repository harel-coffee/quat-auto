#!/usr/bin/env python3
"""
Collected python methods for building up automated ffmpeg commands
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
import shutil
import json

from quat.log import *
from quat.utils.system import shell_call

# some general defined exceptions
class MissingCodecImplementationException(Exception): pass
class FFmpegMissingError(Exception): pass
class FFprobeMissingException(Exception): pass
class NoValidVideoFileException(Exception): pass


def generic_two_pass(inputfile, bitrate, resolution, framerate, outputfilename, profile, logfile, ffmpeg_base_command, codecsettings):
    """ a general two pass command building, e.g. for vp9 and h264 usable """
    first_pass = ffmpeg_base_command.format(inputfile=inputfile,
                                            codecsettings="-pass 1 -passlogfile {} ".format(logfile) + codecsettings,
                                            bitrate=bitrate,
                                            resolution=resolution,
                                            framerate=framerate,
                                            outputfilename=outputfilename)

    second_pass = ffmpeg_base_command.format(inputfile=inputfile,
                                             codecsettings="-pass 2 -passlogfile {} ".format(logfile) + codecsettings,
                                             bitrate=bitrate,
                                             resolution=resolution,
                                             framerate=framerate,
                                             outputfilename=outputfilename)
    return first_pass +  " && " +  second_pass


def h264_encoding(inputfile, bitrate, resolution, framerate, outputfilename, profile, passes, logfile, ffmpeg_base_command):
    """ h264 encoding """
    codecsettings = "-preset slow -c:v h264 -profile:v {profile}".format(profile=profile)

    if passes == 1:
        return ffmpeg_base_command.format(inputfile=inputfile,
                                          codecsettings=codecsettings,
                                          bitrate=bitrate,
                                          resolution=resolution,
                                          framerate=framerate,
                                          outputfilename=outputfilename)


    return generic_two_pass(inputfile, bitrate, resolution, framerate, outputfilename, profile, logfile, ffmpeg_base_command, codecsettings)


def hevc_encoding(inputfile, bitrate, resolution, framerate, outputfilename, profile, passes, logfile, ffmpeg_base_command):
    """ specific handling of hevc encoding settings """
    codecsettings = "-preset slow -c:v hevc "
    # FIXME(stg7) + "-x265-params \"profile={}\"".format(profile) should force profile, but does not change profile
    if passes == 1:
        return ffmpeg_base_command.format(inputfile=inputfile,
                                          codecsettings=codecsettings,
                                          bitrate=bitrate,
                                          resolution=resolution,
                                          framerate=framerate,
                                          outputfilename=outputfilename)

    first_pass = ffmpeg_base_command.format(inputfile=inputfile,
                                            codecsettings=codecsettings + "-x265-params \"pass=1:stats={}\"".format(logfile),
                                            bitrate=bitrate,
                                            resolution=resolution,
                                            framerate=framerate,
                                            outputfilename=outputfilename)

    second_pass = ffmpeg_base_command.format(inputfile=inputfile,
                                             codecsettings=codecsettings + "-x265-params \"pass=2:stats={}\"".format(logfile),
                                             bitrate=bitrate,
                                             resolution=resolution,
                                             framerate=framerate,
                                             outputfilename=outputfilename)
    return first_pass +  " && " +  second_pass


def vp9_encoding(inputfile, bitrate, resolution, framerate, outputfilename, profile, passes, logfile, ffmpeg_base_command):
    """ generate settings for vp9 encoding """
    """
    -quality    May be set to good, best, or realtime
    -speed  This parameter has different meanings depending upon whether quality is set to good or realtime.
            Good valid values are 0-5, with 0 being the highest quality and 5 being the lowest.
            Realtime valid values are 0-15; lower numbers mean higher quality.
    """
    codecsettings = "-quality good -speed 3 -c:v vp9 -profile:v {profile}".format(profile=profile)
    if passes == 1:

        return ffmpeg_base_command.format(inputfile=inputfile,
                                          codecsettings=codecsettings,
                                          bitrate=bitrate,
                                          resolution=resolution,
                                          framerate=framerate,
                                          outputfilename=outputfilename)

    return generic_two_pass(inputfile, bitrate, resolution, framerate, outputfilename, profile, logfile, ffmpeg_base_command, codecsettings)


def build_ffmpeg_command(pvs, output_dir):
    """
    Creates strings out of PVS information for running a ffmpeg instances in a separate shell
    :param pvs: complete PVS (= dict)
    :return: return built string
    """

    # Read metadata from files
    src_lst = os.path.splitext(os.path.basename(pvs["src"]))[0].split("-")

    if shutil.which("ffmpeg") is None:
        raise FFmpegMissingError("you do not have ffmpeg installed, please find a creative way for getting it started.")

    ffmpeg_base_command = ["ffmpeg",
                           "-hide_banner",  # display less information when ffmpeg initialises
                           "-loglevel error",  # show only ffmpeg errors
                           "-y",  # auto overwrite if file exists
                           "-i {inputfile}",  # input video file name
                           "{codecsettings}",  # codec settings
                           "-b:v {bitrate}k",  # used video bitrate
                           "-vf scale=-2:{resolution}",  # rescale to specific resolution
                           "-r {framerate}",  # defined framerate
                           "-an",  # remove audio streams, replace with "-c:a copy" to copy it
                           "{outputfilename}"  # name of output file
                          ]


    implemented_codecs = {"h264": h264_encoding,
                          "hevc": hevc_encoding,
                          "vp9": vp9_encoding
                         }

    required_codec = pvs["profile"][0]

    if required_codec not in implemented_codecs:
        raise MissingCodecImplementationException("your selected codec {} is not implemented so far, add it or cry".format(required_codec))

    basefilename = "-".join(map(str, [src_lst[0],
                                      pvs["profile"][0],
                                      pvs["profile"][1],
                                      str(pvs["targetBitrate"]) + "kbps",
                                      str(pvs["fps"]) + "fps",
                                      pvs["resolution"] + "p",
                                      pvs["passes"]
                                     ])) + ".mkv"  # out filename + container type (e,g. `Beauty-h264-baseline-3750kbps-60fps-720p.mkv`)

    output_filename = output_dir + "/" + basefilename

    logfile = "./logs/" + output_filename.replace("/", "_").replace(".", "_") + "_log.log"

    ff_command = implemented_codecs[required_codec](inputfile=pvs["src"] ,
                                                    bitrate=pvs["targetBitrate"],
                                                    resolution=pvs["resolution"],
                                                    framerate=pvs["fps"],
                                                    outputfilename=output_filename,
                                                    profile=pvs["profile"][1],
                                                    passes=pvs["passes"],
                                                    logfile=logfile,
                                                    ffmpeg_base_command=" ".join(ffmpeg_base_command))
    return ff_command


def build_commands_to_convert_to_4k_and_lossless(processes_video_dir, lossless_output_dir):
    """ read all files in processes_video_dir and create commands to convert
        to 4k resolution and lossless codec
    """
    # global settings for each lossless encoded sequence,
    # TODO(stg7) needs to be modified manually find a better way
    settings = {"pixel_format": "yuv422p10le",
                "infile": "infile",
                "scale": "-2:2160",
                "outputfile": "outputfile",
                "codec": "ffvhuff"
               }

    base_cmd = """ffmpeg -i "{infile}" -y -c:v {codec} -vf scale={scale} -pix_fmt {pixel_format} "{outputfile}" -pix_fmt {pixel_format}"""

    os.makedirs(lossless_output_dir, exist_ok=True)

    commands = []
    for infile in utils.get_files(processes_video_dir):
        settings["infile"] = infile
        settings["outputfile"] = infile.replace(processes_video_dir, lossless_output_dir)
        commands.append(base_cmd.format(**settings))

    return commands


def analyze_video_file(filename):
    """ run ffprobe to get some information of a given video file
    """
    if shutil.which("ffprobe") is None:
        raise FFprobeMissingException("you need to have ffprobe installed, please read README.md.")

    if not os.path.isfile(filename):
        raise NoValidVideoFileException("{} is not a valid file".format(filename))

    cmd = "ffprobe -show_format -select_streams v:0 -show_streams -of json {filename} 2>/dev/null".format(filename=filename)

    res = shell_call(cmd).strip()

    if len(res) == 0:
        raise Exception("{} is somehow not valid, so ffprobe could not extract anything".format(filename))

    res = json.loads(res)

    needed = {"pix_fmt": "unknown",
              "bits_per_raw_sample": "unknown",
              "width": "unknown",
              "height": "unknown",
              "avg_frame_rate": "unknown"
             }
    for stream in res["streams"]:
        for n in needed:
            if n in stream:
                needed[n] = stream[n]
                if n == "avg_frame_rate":  # convert framerate to numeric integer value
                    needed[n] = round(eval(needed[n]))
    needed["target_bitrate"] = res.get("format", {}).get("bit_rate", -1)
    needed["processed_video_sequence"] = os.path.basename(filename)
    return needed


if __name__ == "__main__":
    print("this is just a lib")
