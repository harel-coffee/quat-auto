#!/usr/bin/env python3
"""
Logging helpers

lInfo, lWarn, lError
--------------------
.. code-block:: python

    from quat.log import *
    lInfo("hello")

will printout a info log message

jprint, jPrint
--------------
.. code-block:: python

    from quat.log import *
    jPrint({"my": "json"})

will printout a json formatted version of the object

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

formatter = logging.Formatter(
    fmt='%(levelname)s: %(message)s'
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

# \033[1;30m - black
# \033[1;31m - red
# \033[1;32m - green
# \033[1;33m - yellow
# \033[1;34m - blue
# \033[1;35m - magenta
# \033[1;36m - cyan
# \033[1;37m - white

logging.addLevelName(logging.CRITICAL, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.CRITICAL))
logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
logging.addLevelName(logging.DEBUG, "\033[1;35m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
logging.basicConfig(level=logging.ERROR)


LOGGING_LEVEL = logging.DEBUG

def lInfo(msg):
    """prints `msg` as info log message"""
    _logger = logging.getLogger(__name__)
    _logger.setLevel(LOGGING_LEVEL)
    _logger.info(msg)


def lError(msg):
    """prints `msg` as error log message"""
    _logger = logging.getLogger(__name__)
    _logger.setLevel(LOGGING_LEVEL)
    _logger.error(msg)


def lDbg(msg):
    """prints `msg` as debug log message"""
    _logger = logging.getLogger(__name__)
    _logger.setLevel(LOGGING_LEVEL)
    _logger.debug(msg)


def lWarn(msg):
    """prints `msg` as warning log message"""
    _logger = logging.getLogger(__name__)
    _logger.setLevel(LOGGING_LEVEL)
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