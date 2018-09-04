#!/usr/bin/env python3

import datetime
import time

from .log import *


def timeit(func):
    """
    get the required time for a function call on as infor on stdout
    """
    def function_wrapper(*args, **kwargs):
        start_time = time.time()
        lInfo("start {}".format(func.__name__))
        res = func(*args, **kwargs)
        overall_time = time.time() - start_time
        lInfo("calculation done for {}: {} s; {}".format(func.__name__, overall_time, str(datetime.timedelta(seconds=overall_time))))
        return res
    return function_wrapper


def p_bar(iterable, total=100):
    """ progress bar """
    step = 0
    progress = 0
    print('\r' + colorred("[CALC ] ") + '[{0}] {1}%'.format('#' * (progress // 5), progress), end="")

    for i in iterable:
        progress = int(100 * step / total)
        char = "/" if step % 2 == 0 else "\\"
        print('\r' + colorred("[CALC{0}] ".format(char)) + '[{0}] {1}%'.format('#' * (progress // 5), progress), end="")
        yield i
        step += 1

    progress = 100
    print('\r' + colorred("[CALC ] ") + '[{0}] {1}%'.format('#' * (progress // 5), progress))


def progress_bar(iterable, max=100):
    """ run progress bar on iterable """
    results = []
    for res in p_bar(iterable, max):
        yield res


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    from: https://stackoverflow.com/a/16915734
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def align_vectors(x, y):
    """
    align the length of two align_vectors
    """
    min_len = min(len(x), len(y))
    x = x[0:min_len]
    y = y[0:min_len]
    return x, y


