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
"""
    Some collected and hopefully usefule methods for file handling
    Author: Steve Göring
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
