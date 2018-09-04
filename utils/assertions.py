#!/usr/bin/env python3
import os
import sys

from ..log import *


def msg_assert(cond, message_error="", message_ok=""):
    if not cond:
        lError(message_error)
        assert(cond)
    if message_ok != "":
        lInfo(message_ok)


def assert_file(file):
    if not os.path.isfile(file):
        lError("{} does not exists".format(file))
        return
    lInfo("{} exists.".format(file))


def json_assert(json, required_keys):
    if type(required_keys) is list:
        for required_key in required_keys:
            json_assert(json, required_key)
        return
    # required_keys is a single string
    if required_keys not in json:
        llError("{} is not in json object, but this key is required".format(required_keys))
        sys.exit(-1)