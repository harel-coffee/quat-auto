#!/usr/bin/python3
"""
Script automatic detection of cuts in a video stream

Note: Tuned for big bucks bunny

Authors: Serge Molina, Steve Göring
"""

import sys
import os
import math
import argparse

import multiprocessing
from multiprocessing import Pool

from quat.log import *
from quat.utils.system import shell_call

import numpy as np
import cv2


def cuts_iterator(video_filename):
    '''
    Function that takes an opencv video stream and yields cuts timing when detected
    The cuts detection is based on detection of peaks in the standard
    deviation of two successive frames

    TODO: some parts can be replaced by cuts-feature inside quat
    '''
    cap = cv2.VideoCapture(video_filename)

    fps = cap.get(cv2.CAP_PROP_FPS)

    last_image = None
    frame_count = 0
    previous_diff_weighted = 0
    if not cap.isOpened():
        print("[error] video {} could not be opended".format(video_filename))
        sys.exit(-1)

    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Scaling down the image speeds up computations and reduces false positives
        image = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_LANCZOS4)

        # Converting the image type to a 16 bits signed integers image prevents over/under flows
        image = np.int16(image)

        time_sec = frame_count / fps
        # TODO(stg7): this does not work for hours
        minutes_hr = int(time_sec / 60)
        secondes_hr = time_sec % 60

        if last_image is not None:
            current_diff = np.std((image - last_image).ravel())

            if current_diff > 30 and current_diff > 4 * previous_diff_weighted:
                yield (frame_count, minutes_hr, secondes_hr, time_sec * 1000)

            previous_diff_weighted = previous_diff_weighted * 0.5 + 0.5 * current_diff

        last_image = image
        frame_count += 1

    yield (frame_count, minutes_hr, secondes_hr, time_sec * 1000)


def extract_cuts(video_filename, min_cut_duration, output_dir, cmd_filename, cpu_count):
    """ extract cuts for one video """
    lInfo("start analyzing {}".format(video_filename))
    if not os.path.isfile(video_filename):
        print("[error] {} is not a valid file".format(video_filename))
        return

    last_cut = "00:00:00"
    last_time_ms = 0

    cmd = """ffmpeg -i {infile} -y -c copy -ss {start} -to {end} {outfile} """

    commands = []
    largest_cut_duration = 0
    # Iteration over the cuts
    i = 0
    for _, minutes_hr, secondes_hr, time_ms in cuts_iterator(video_filename):

        # TODO: extension to hours?
        curr_cut = "00:" + str(math.floor(minutes_hr)).zfill(2) + ":" + str(math.floor(secondes_hr)).zfill(2)

        # TODO: maybe just change time_ms to seconds instead of ms
        if time_ms - last_time_ms >= 1000 * min_cut_duration:
            # found a suitable cut
            lInfo("found a cut from {} to {}, duration {}s".format(last_cut, curr_cut, (time_ms - last_time_ms) / 1000))
            outfile = output_dir + "/" + os.path.splitext(os.path.basename(video_filename))[0] + "_" + str(i) + ".mkv"
            commands.append(cmd.format(infile=video_filename,
                                       start=last_cut,
                                       end=curr_cut,
                                       outfile=outfile))
            i += 1
        largest_cut_duration = max(time_ms - last_time_ms, largest_cut_duration)
        last_time_ms = time_ms
        last_cut = "00:" + str(math.ceil(minutes_hr)).zfill(2) + ":" + str(math.ceil(secondes_hr)).zfill(2)

    lInfo("write commands to {}".format(cmd_filename))
    with open(cmd_filename, "a") as cmd_file:
        cmd_file.write("\n".join(commands + ["# end of {}\n".format(video_filename)]))

    lInfo("extract {} sequences via multiprocessing and store in {}".format(len(commands), output_dir))
    pool = Pool(processes=cpu_count)
    pool.map(shell_call, commands)
    lInfo("done")


def main(_):
    """ extracts cuts of a video file """

    def not_neg_int(maybe_number):
        """ checks if `maybe_number` is a not negative integer """
        num = int(maybe_number)
        if num < 0:
            raise argparse.ArgumentTypeError("a not negative integer is required.")
        return num

    parser = argparse.ArgumentParser(
        description='cuts detector',
        epilog='2017',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'video_filename',
        type=str,
        nargs="+",
        help="filename of input video"
    )
    parser.add_argument(
        '--min_cut_duration',
        type=not_neg_int,
        default=10,
        help="minimum duration of an extracted cut; set to 0 for disabling"
    )
    parser.add_argument(
        '--cpu_count',
        type=int,
        default=multiprocessing.cpu_count(),
        help="cpus/threads that are used for processing"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./results",
        help="output_directory for storing calculated features"
    )
    parser.add_argument(
        '--cmd_filename',
        type=str,
        default="commands.list",
        help="file where all ffmpeg commands are stored"
    )

    argsdict = vars(parser.parse_args())

    os.makedirs(argsdict["output_dir"], exist_ok=True)

    for video_filename in argsdict["video_filename"]:
        extract_cuts(
            video_filename,
            argsdict["min_cut_duration"],
            argsdict["output_dir"],
            argsdict["cmd_filename"],
            argsdict["cpu_count"]
        )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
