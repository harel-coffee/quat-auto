#!/usr/bin/env python3
"""
    Copyright 2016-today
    Some collected and hopefully usefule methods for file handling

    Author: Steve Göring, Konstantin Brand

"""

import os
import bz2
import gzip
import json


def get_files_by_extension(base_dir, extension):
    """ get list of files in a base_dir with specified extension
    """
    return list(filter(lambda x: extension in x,
                       list(map(lambda x: base_dir + "/" + x,
                                os.listdir(base_dir)))))


def get_files_by_extensions(base_dir, extensions=[""]):
    """ get files by a list of extensions """
    res = []
    for ext in extensions:
        res += get_files_by_extension(base_dir, ext)
    return res


def file_open(filename, mode="r"):
    """ Open a file, and if you add bz2 to filename a compressed file will be opened
    """
    if "bz2" in filename:
        return bz2.open(filename, mode + "t")
    if "gz" in filename:
        return gzip.open(filename, mode + "t")
    return open(filename, mode)


def get_filename_without_extension(full_filename):
    """
    extract the plain filename without basename and extension of a given path and full filename
    """
    return os.path.splitext(os.path.basename(full_filename))[0]


def read_file(file_name):
    """
    read a text file into a string
    :file_name file to open
    :return content as string
    """
    f = open(file_name, "r")
    content = "".join(f.readlines())
    f.close()
    return content


def read_json(filename):
    with open(filename) as fobj:
        j = json.load(fobj)
    return j


def write_json(j, filename):
    with open(filename, "w") as file:
        json.dump(j, file)


if __name__ == "__main__":
    print("[error] this is just a lib")
