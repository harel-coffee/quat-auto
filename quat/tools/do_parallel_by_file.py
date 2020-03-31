#!/usr/bin/env python3
"""
tool for running scripts parallel

example usage:
$ ./do_parallel_by_file.py commands.list

where in commands.list all commandas are stored

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

import re
import sys
import os
import argparse
from multiprocessing import Pool
import multiprocessing

from quat.log import *


def do_it(command):
    lInfo("run {}".format(command))
    res = os.system(command)
    if res != 0:
        return "error with: {}".format(command)
    lInfo("done {}".format(command))
    return command


def main(params=[]):
    parser = argparse.ArgumentParser(
        description="run a command on several files parallel",
        epilog="stg7 2016",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=multiprocessing.cpu_count(),
        help="thread/cpu count",
    )
    parser.add_argument(
        "infile",
        type=str,
        help="inputfile where all commands are stored in each line that should run parallel",
    )
    argsdict = vars(parser.parse_args())

    cpu_count = argsdict["cpu_count"]
    lInfo("running with " + str(cpu_count) + " threads")

    commands = []
    with open(argsdict["infile"]) as commands_file:
        commands = [x.strip() for x in commands_file.readlines()]

    pool = Pool(processes=cpu_count)
    res = pool.map(do_it, commands)
    print("commands:")
    print("\n".join(res))

    lInfo("done.")


if __name__ == "__main__":
    main(sys.argv[1:])
