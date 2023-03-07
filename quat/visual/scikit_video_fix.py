#!/usr/bin/env python3
"""
fix for scikit-video issues
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
import numpy as np

# ugly hack to bypass numpy issues with old scikit-video version, which hopefully will be updated in 2023, see https://github.com/scikit-video/scikit-video/pull/169
np.int = int
np.float = float
np.bool = bool


def psnr(referenceVideoData, distortedVideoData, bitdepth=8):
    """ a minimal adjusted variant of psnr from scikit video """
    bitdepth = np.int64(bitdepth)
    maxvalue = np.int64(2**bitdepth - 1)
    maxsq = maxvalue**2

    referenceFrame = referenceVideoData.astype(np.float64)
    distortedFrame = distortedVideoData.astype(np.float64)

    mse = np.mean((referenceFrame - distortedFrame)**2)
    psnr = 10 * np.log10(maxsq / mse)
    return psnr
