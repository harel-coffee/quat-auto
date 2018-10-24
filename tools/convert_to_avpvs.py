#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import pandas as pd

import yaml

def convert_to_avpvs(input_file, tmp_folder, src_framerate = "60/1"):

    if "_" in input_file and "P2" in input_file:
        # its probably a PNATS DB
        print("its a pnats db")
        tmp = os.path.basename(input_file).split("_", 2)
        database, srcid = tmp[0], tmp[1]
        #print(database, srcid)
        if os.path.isfile("all_src_framerates.csv"):
            # only for training
            fps = pd.read_csv("all_src_framerates.csv")
            fps = fps[(fps["database"] == database) & (fps["srd_id"] == srcid)]
            if len(fps) == 1:
                src_framerate = fps["fps"].values[0]
        else:

            dbpath = os.path.dirname(input_file).replace("videoSegments", "").replace("avpvs", "").replace("srcVid", "")
            yaml_file = list(glob.glob(dbpath + "/*_restricted.yaml"))[0]

            with open(yaml_file) as ry:
                y = yaml.load(ry)

            for x in y["pvsInfo"]:
                db, src, hrc = x.split("_", 2)

                fps = y["pvsInfo"][x]["avpvsFramerate"]
                pxfmt = y["pvsInfo"][x]["avpvsPixFmt"]
                if srcid == src:
                    break
            print(src, fps, pxfmt)

    # rescale case
    avpvs_width = 3840
    avpvs_height = -2

    os.makedirs(tmp_folder, exist_ok=True)

    output_file = os.path.join(tmp_folder, os.path.splitext(os.path.basename(input_file))[0] + ".mkv")
    cmd = """
    ffmpeg -nostdin -loglevel quiet
    -y
    -i {input_file}
    -filter:v scale={avpvs_width}:{avpvs_height},fps={src_framerate},setsar=1/1
    -c:v ffvhuff -an
    {output_file} 2>/dev/null""".format(**locals())

    # remove multiple spaces
    cmd = " ".join(cmd.split())
    print(f"convert to avpvs: {input_file}")
    ret = os.system(cmd)
    if ret != 0:
        sys.exit(-1)
    return output_file


def crop_video(input_file, tmp_folder):
    os.makedirs(tmp_folder, exist_ok=True)

    output_file = os.path.join(
        tmp_folder,
        os.path.splitext(os.path.basename(input_file))[0] + "_cropped.mkv"
    )
    cmd = """
    ffmpeg -nostdin -loglevel quiet
    -y
    -i {input_file}
    -filter:v crop=360*in_w/in_h:360
    -c:v ffvhuff -an
    {output_file} 2>/dev/null""".format(**locals())

    # remove multiple spaces
    cmd = " ".join(cmd.split())
    print(f"crop video: {input_file}")
    ret = os.system(cmd)
    if ret != 0:
        sys.exit(-1)
    return output_file


def rescale_video(input_file, tmp_folder):
    os.makedirs(tmp_folder, exist_ok=True)

    output_file = os.path.join(
        tmp_folder,
        os.path.splitext(os.path.basename(input_file))[0] + "_rescaled.mkv"
    )
    cmd = """
    ffmpeg -nostdin -loglevel quiet
    -y
    -i {input_file}
    -filter:v scale=-2:360
    -c:v ffvhuff -an
    {output_file}  2>/dev/null""".format(**locals())

    # remove multiple spaces
    cmd = " ".join(cmd.split())
    print(f"rescale video: {input_file}")
    ret = os.system(cmd)
    if ret != 0:
        sys.exit(-1)
    return output_file


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(description='convert video segment to avpvs',
                                     epilog="stg7 2018",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, help="video segment")
    parser.add_argument("--tmp_folder", type=str, default="./tmp", help="folder for storing the avpvs file")

    a = vars(parser.parse_args())
    convert_to_avpvs(a["video"], a["tmp_folder"])