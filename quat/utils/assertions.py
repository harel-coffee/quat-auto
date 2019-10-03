#!/usr/bin/env python3
"""
Collection of assertions helper.
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

from ..log import *


def msg_assert(cond, message_error="", message_ok=""):
    """
    assert with a message for error or no error

    Parameters
    ----------
    cond : bool
        condition to check
    message_error : str
        message that is shown in case of error
    message_ok : str
        message that is shown in case of ok, or if empty nothing is shown
    """
    if not cond:
        lError(message_error)
        assert cond
    if message_ok != "":
        lInfo(message_ok)


def assert_file(file, withassert=False):
    """ checks if a file is existing
    """
    if not os.path.isfile(file):
        lError("{} does not exists".format(file))
        if withassert:
            assert False
        return
    lInfo("{} exists.".format(file))


def json_assert(json, required_keys):
    """ checks if required_keys are present in json object """
    if type(required_keys) is list:
        for required_key in required_keys:
            json_assert(json, required_key)
        return
    # required_keys is a single string
    if required_keys not in json:
        llError(
            "{} is not in json object, but this key is required".format(required_keys)
        )
        assert False
