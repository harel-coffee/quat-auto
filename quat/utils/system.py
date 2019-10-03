#!/usr/bin/env python3
"""
Collection of system helper functions.

TODO: unify documentation (e.g. parameters)
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
import subprocess
import glob


def shell_call(call):
    """
    Run a program via system call and return stdout + stderr.
    @param call programm and command line parameter list, e.g shell_call("ls /")
    @return stdout and stderr of programm call
    """
    try:
        output = subprocess.check_output(call, universal_newlines=True, shell=True)
    except Exception as e:
        output = str(e.output)
    return output


def get_shebang(file_name):
    """
    read shebang of a file
    :file_name file to open
    :return shebang
    """
    try:
        f = open(file_name, "r")
        lines = f.readlines()
        f.close()
        if len(lines) == 0:
            return ""
        shebang = lines[0]
        return shebang
    except Exception as e:
        return ""


def create_dir_if_not_exists(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)


def get_prog_name():
    """
    return name of script, that was called on command line
    """
    return os.path.basename(sys.argv[0])


def lglob(path_pattern):
    """
    returns list of glob results based on the given path_pattern
    """
    return list(glob.glob(path_pattern))


if __name__ == "__main__":
    print("\033[91m[ERROR]\033[0m lib is not a standalone module")
    exit(-1)
