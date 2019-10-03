#!/usr/bin/env python3
"""
Logging helpers
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

import logging
import json

formatter = logging.Formatter(fmt="%(levelname)s: %(message)s")

handler = logging.StreamHandler()
handler.setFormatter(formatter)

# TODO: think about a better way
color_codes = {
    "black": "\033[1;30m",
    "red": "\033[1;31m",
    "green": "\033[1;32m",
    "yellow": "\033[1;33m",
    "blue": "\033[1;34m",
    "magenta": "\033[1;35m",
    "cyan": "\033[1;36m",
    "white": "\033[1;37m",
    "end_code": "\033[1;0m",
}


logging.addLevelName(
    logging.CRITICAL,
    color_codes["red"]
    + logging.getLevelName(logging.CRITICAL)
    + color_codes["end_code"],
)
logging.addLevelName(
    logging.ERROR,
    color_codes["red"] + logging.getLevelName(logging.ERROR) + color_codes["end_code"],
)
logging.addLevelName(
    logging.WARNING,
    color_codes["yellow"]
    + logging.getLevelName(logging.WARNING)
    + color_codes["end_code"],
)
logging.addLevelName(
    logging.INFO,
    color_codes["green"] + logging.getLevelName(logging.INFO) + color_codes["end_code"],
)
logging.addLevelName(
    logging.DEBUG,
    color_codes["blue"] + logging.getLevelName(logging.DEBUG) + color_codes["end_code"],
)
logging.basicConfig(level=logging.ERROR)


LOGGING_LEVEL = logging.DEBUG

_logger = logging.getLogger(__name__)
_logger.setLevel(LOGGING_LEVEL)


def lInfo(msg):
    """prints `msg` as info log message"""
    _logger.info(msg)


def lError(msg):
    """prints `msg` as error log message"""
    _logger.error(msg)


def lDbg(msg):
    """prints `msg` as debug log message"""
    _logger.debug(msg)


def lWarn(msg):
    """prints `msg` as warning log message"""
    _logger.warning(msg)


def colorgreen(m):
    """
    return m with colored green code, only for linux
    """
    return "\033[92m" + m + "\033[0m"


def colorred(m):
    """
    return m with colored red code, only for linux
    """
    return "\033[91m" + m + "\033[0m"


def jprint(x):
    """
    prints an object `x` as json formatted to stdout
    """
    print(json.dumps(x, indent=4, sort_keys=True))


def jPrint(x):
    """
    renamed version of jprint
    """
    jprint(x)
