#!/usr/bin/env python3
"""
Helpers for parallel processing
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

from multiprocessing import Pool
import multiprocessing


def run_parallel_by_grouping(items, function, arguments, num_cpus=8):
    """
    devides items to num_cpus groups and
    applies function with each group + arguments in parallel threads
    """
    items = sorted(items)
    # build group of items that are matching number of cpus
    groups = []
    length = len(items) // multiprocessing.cpu_count()
    for x in range(0, len(items), length):
        groups.append(items[x: x + length])
    # add for each group the shared arguments
    params = [tuple([item_group] + list(arguments)) for item_group in groups]
    # run parallel
    pool = Pool(processes=num_cpus)
    return pool.starmap(function, params)


def run_parallel(items, function, arguments, num_cpus=8):
    """
    run a function call parallel, for each item
    """
    params = [tuple([i] + arguments) for i in items]
    # run parallel
    pool = Pool(processes=num_cpus)
    res =  pool.starmap(function, params)
    return res

