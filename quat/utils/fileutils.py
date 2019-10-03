#!/usr/bin/env python3
"""
Some collected and hopefully usefule methods for file handling
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
import bz2
import gzip
import json


def get_files_by_extension(base_dir, extension):
    """
    get list of files in a base_dir with specified extension

    Parameters
    ----------
    base_dir : str
        dictionary to search
    extension: str
        extension to check

    Returns
    -------
    list of files matching the extension

    """
    return list(
        filter(
            lambda x: extension in x,
            list(map(lambda x: base_dir + "/" + x, os.listdir(base_dir))),
        )
    )


def get_files_by_extensions(base_dir, extensions=[""]):
    """ get files by a list of extensions, see get_files_by_extension """
    res = []
    for ext in extensions:
        res += get_files_by_extension(base_dir, ext)
    return res


def file_open(filename, mode="r"):
    """ Open a file (depending on the mode), and if you add bz2 or gz to filename a compressed file will be opened,
    file_open can replace a typical with open(filename) statement
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

    Parameters
    ----------
    file_name : str
        file to open

    Returns
    -------
    return content of the complete file as string
    """
    content = ""
    with open(file_name, "r") as fp:
        content = "".join(fp.readlines())
    return content


def read_json(filename):
    """ reads a json file """
    with open(filename) as fobj:
        j = json.load(fobj)
    return j


def write_json(j, filename, prettify=False):
    """ writes a json object j to a file """
    indent = 4 if prettify else None
    sort_keys = prettify
    with open(filename, "w") as file:
        json.dump(j, file, indent=indent, sort_keys=sort_keys)


if __name__ == "__main__":
    print("[error] this is just a lib")
