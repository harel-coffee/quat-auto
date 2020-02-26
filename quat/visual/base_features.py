#!/usr/bin/env python3
"""
Video and image no-reference based features.
Base features can also be applied for full-ref calculations,
using the `calc_ref_dis` method.
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
import cv2
import os
import json
import copy
from abc import ABC, abstractmethod

import numpy as np
import skimage.color
import skimage.io
import skvideo.motion
import skvideo
from skvideo.measure.strred import extract_info as strred_extract_info
from skimage import img_as_ubyte
import scipy
from scipy import ndimage


from ..log import *


class Feature:
    """
    abstract base class for all features,
    handles automatic storage and loading of calculated feature values
    """

    def __init__(self):
        self._values = []

    @abstractmethod
    def calc(self, frame):
        """ perform feature calculation for a single frame """
        pass

    def calc_ref_dis(self, dframe, rframe):
        """ performs a full-ref style calculation,
        where the resulting features are calculated on both frames,
        and further difference values are stored,

        Parameters
        ----------
        dframe : 3d array
            distorted video frame
        rframe : 3d array
            reference video frame

        Returns
        -------
        a dict {"diff": values, "ref": values, "dis": values} or
        dict {"diff_" + k: values, "ref_" + k: values, "dis_" + k: values} for all keys `k` in the underlying feature.
        """
        # TODO: rename calc_ref_dis --> calc_dis_ref
        # this creates for each feature stream a copy instance of the used feature
        # TODO: prettify the next code part
        if not hasattr(self, "_ref_instance"):
            try:
                self._ref_instance = copy.deepcopy(self)
            except:
                # TODO: fix to handle MovementFeatures
                self._ref_instance = self.__class__()
                lWarn(
                    f"please check if {self.__name__} does not require parameters for __init__() call"
                )
        if not hasattr(self, "_dis_instance"):
            try:
                self._dis_instance = copy.deepcopy(self)
            except:
                # TODO: fix to handle MovementFeatures
                self._dis_instance = self.__class__()

        v1 = self._ref_instance.calc(rframe)
        v2 = self._dis_instance.calc(dframe)

        res = {}
        if type(v1) == dict:
            for k in v1:
                res["diff_" + k] = v1[k] - v2[k]
                res["dis_" + k] = v1[k]
                res["ref_" + k] = v2[k]
        elif type(v1) == list:
            res["diff"] = np.array(v1) - np.array(v2)
            res["dis"] = v1
            res["ref"] = v2
        else:
            res["diff"] = v1 - v2
            res["dis"] = v1
            res["ref"] = v2
        self._values.append(res)
        return res

    def calc_dis_ref(self, dframe, rframe):
        """ fix for consistent naming scheme """
        return self.calc_ref_dis(dframe, rframe)

    def get_values(self):
        """ returns all stored feature values """
        return self._values

    def fullref(self):
        """ used to check if it is a full reference feature """
        return False

    def _feature_filename(self, folder, video, name):
        """ generates a feature filename for a given `video`
        for a specific feature folder `folder` and
        adds a feature name `name`
        """
        bn = os.path.basename(os.path.splitext(video)[0])
        if name == "":
            name = self.__class__.__name__
        rfn = os.path.join(folder, bn + "_" + name + ".json")
        return rfn

    def load(self, folder, video, name=""):
        """ loads a feature from a feature folder `folder`,
        feature filename is estimated using the _feature_filename
        """
        os.makedirs(folder, exist_ok=True)
        fn = self._feature_filename(folder, video, name)
        if os.path.isfile(fn):
            with open(fn) as ffp:
                try:
                    j = json.load(ffp)
                except:
                    lWarn(
                        f"there is something wrong with {video}, feature: {name}, re-calcuation performed"
                    )
                    # loading of feature value is not possible,
                    # force to calculate it again
                    return False
                self._values = j["values"]
                return j["values"]
        return False

    def store(self, folder, video, name=""):
        """ stores a feature to a feature folder `folder`,
        feature filename is estimated using the _feature_filename
        """
        os.makedirs(folder, exist_ok=True)
        fn = self._feature_filename(folder, video, name)
        v = {"name": name, "values": self._values, "video": video}
        with open(fn, "w") as ffp:
            json.dump(v, ffp, indent=4, sort_keys=True)
        return fn


class MovementFeatures(Feature):
    """ Calculates movement feature, using background removement,
    based on master thesis of julian zebelein
    """

    def __init__(self):
        self._fgbg = cv2.createBackgroundSubtractorMOG2()
        self._values = []

    def calc(self, frame, debug=False):
        file_height = frame.shape[0]
        file_width = frame.shape[1]
        frame = img_as_ubyte(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        final = cv2.bilateralFilter(gray, 9, 75, 75)
        fgmask = self._fgbg.apply(final)
        # final = cv2.bilateralFilter(fgmask,9,75,75)
        if debug:
            cv2.imshow(
                "o", cv2.resize(frame, (600, int(600 * file_height / file_width)))
            )
            cv2.imshow(
                "fg", cv2.resize(fgmask, (600, int(600 * file_height / file_width)))
            )
            k = cv2.waitKey(30) & 0xFF

        non_zero = cv2.countNonZero(fgmask)
        moving_percentage = (non_zero * 100) / (file_height * file_width)
        value = float(moving_percentage)
        self._values.append(value)
        return value

    def get_values(self):
        if len(self._values) > 0:
            r = 0
            if type(self._values[0]) == dict:
                r = self._values[0]
                for k in r:
                    r[k] = 0
            self._values[0] = r
        return self._values


class CutDetectionFeatures(Feature):
    """ Estimates scene cuts of a given video (approximation),
    implemented by Serge Molina
    """

    def __init__(self):
        self._values = []
        self._last_frame = None
        self._previous_diff_weighted = 0

    def calc(self, frame):
        # TODO change to skiimage
        # Scaling down the image speeds up computations and reduces false positives
        image = cv2.resize(frame, dsize=(320, 240), interpolation=cv2.INTER_LANCZOS4)
        cut = 0
        # Converting the image type to a 16 bits signed integers image prevents over/under flows
        image = np.int16(image)

        if self._last_frame is not None:
            current_diff = np.std((image - self._last_frame).ravel())

            if current_diff > 30 and current_diff > 4 * self._previous_diff_weighted:
                # we detect this as a cut
                cut = 1

            self._previous_diff_weighted = (
                self._previous_diff_weighted * 0.5 + 0.5 * current_diff
            )
        self._last_frame = image
        self._values.append(cut)
        return cut


class SiFeatures(Feature):
    """
    Calculates SI values of a video frame
    important: SI values are finally in a 0..1 range, due to float conversion
    """

    def __init__(self):
        self._values = []

    def calc(self, frame):
        def calculate_si(frame_data):
            sobx = ndimage.sobel(frame, axis=0)
            soby = ndimage.sobel(frame, axis=1)
            value = np.hypot(sobx, soby).std()
            return float(value)

        frame = skimage.color.rgb2gray(frame)
        value = calculate_si(frame)
        self._values.append(value)
        return value


class TiFeatures(Feature):
    """
    Calculates TI values
    important: TI values are finally in a 0..1 range, due to float conversion
    """

    def __init__(self):
        self._values = []
        self._previous_frame = None

    def calc(self, frame):
        def calculate_ti(frame_data, previous_frame_data):
            if previous_frame_data is None:
                return 0
            return float((frame_data - previous_frame_data).std())

        frame = skimage.color.rgb2gray(frame)
        value = calculate_ti(frame, self._previous_frame)
        self._previous_frame = frame
        self._values.append(value)
        return value


class TemporalFeatures(Feature):
    """ A temporal feature, using RMSE of consecutive frames,
    somehow similar to TI, but not applied on gray frames
    """

    def __init__(self):
        self._values = []
        self._previous_frame = None

    def rmse(self, x, y):
        return np.sqrt(((x - y) ** 2).mean())

    def calc(self, frame):
        if self._previous_frame is None:
            value = 0
        else:
            value = float(self.rmse(frame.flatten(), self._previous_frame.flatten()))
        self._previous_frame = frame
        self._values.append(value)
        return value


class StrredNoRefFeatures(Feature):
    """
    calculate entropy of subbands, with the feature that is used in strred, however, this feature does not
    consider a reference video, it justs calculates mean of spatial and temporal features of strred
    """

    def __init__(self):
        self._values = []
        self._previous_frame = None

    def calc(self, frame):
        def calculate(frame_data, previous_frame_data):
            if previous_frame_data is None:
                return {
                    "spatial.mean": 0,
                    "spatial.std": 0,
                    "temporal.mean": 0,
                    "temporal.std": 0,
                }
            spatial, temporal = strred_extract_info(previous_frame_data, frame_data)
            return {
                "spatial.mean": float(spatial.mean()),
                "spatial.std": float(spatial.std()),
                "temporal.mean": float(temporal.mean()),
                "temporal.std": float(temporal.std()),
            }

        frame = skimage.color.rgb2gray(frame).astype(np.float32)
        value = calculate(frame, self._previous_frame)
        self._previous_frame = frame
        self._values.append(value)
        return value


class BlockMotion(Feature):
    """
    calculates block motion of two following frames,
    block size is estimated by 5% of the height of the input frame,
    this is done to be resolution independent and faster
    """

    def __init__(self):
        self._values = []
        self._last_frame = None

    def calc(self, frame):
        per_frame_values = {"blkm.zeros": 0, "blkm.ones": 0, "blkm.minusones": 0}
        if self._last_frame is not None:
            videodata = np.array([self._last_frame, frame])
            blocksize = int(self._last_frame.shape[0] * 0.05)
            motion = skvideo.motion.blockMotion(
                videodata, method="SE3SS", mbSize=blocksize
            )
            m = motion[0].flatten()
            blk_motion_zeros = np.count_nonzero(m == 0) / len(m)
            blk_motion_ones = np.count_nonzero(m == 1) / len(m)
            blk_motion_minusones = np.count_nonzero(m == -1) / len(m)
            per_frame_values = {
                "blkm.zeros": blk_motion_zeros,
                "blkm.ones": blk_motion_ones,
                "blkm.minusones": blk_motion_minusones,
            }
            self._values.append(per_frame_values)
        self._last_frame = frame
        return per_frame_values


class CuboidRow(Feature):
    """
    Motion estimation using a window of 60 frames and a cuboid video of the video,
    handles only rows of the frames
    """

    WINDOW = 60  # maximum number of frames in sliding window

    def __init__(self, row):
        """ row specifies the column that should be used in %"""
        self._row = row
        self._rows = []
        self._values = []

    def calc(self, frame):
        frame_gray = skimage.color.rgb2gray(frame)
        if len(self._rows) >= self.WINDOW:
            self._rows = self._rows[1:]
        row_i = int(self._row * (frame.shape[0] - 1))
        tmp = frame_gray[row_i].copy()  # copy reduces memory !
        self._rows.append(tmp)
        v = ndimage.sobel(self._rows).std()
        self._values.append(v)
        return v


class CuboidCol(Feature):
    """
    Motion estimation using a window of 60 frames and a cuboid video of the video,
    handles only columns of the frames
    """

    WINDOW = 60  # maximum number of frames in sliding window

    def __init__(self, col):
        """ col specifies the column that should be used in %"""
        self._col = col
        self._cols = []
        self._values = []

    def calc(self, frame):
        frame_gray = skimage.color.rgb2gray(frame)
        if len(self._cols) >= self.WINDOW:
            self._cols = self._cols[1:]
        col_i = int(self._col * (frame.shape[1] - 1))
        tmp = frame_gray[:, col_i].copy()  # copy reduces memory !
        self._cols.append(tmp)
        v = ndimage.sobel(self._cols).std()
        self._values.append(v)
        return v


class Staticness(Feature):
    """
    calculates how static the video is
    """

    def __init__(self):
        self._values = []
        self._frame_sum = None
        self._frame_no = 1

    def calc(self, frame):
        # convert datatype, required otherwise overflows
        frame = np.array(img_as_ubyte(frame), dtype=np.int64)
        if self._frame_sum is None:
            self._frame_sum = frame
        else:
            self._frame_sum += frame

        sobeled_image = ndimage.sobel(self._frame_sum // self._frame_no)
        self._frame_no += 1
        v = sobeled_image.std()
        self._values.append(v)

        # skimage.io.imshow(sobeled_image)
        # skimage.io.show()

        return v


class UHDSIM2HD(Feature):
    """
    calculates similarity of UHD input resolution to HD,
    if input frame is not UHD resolution, it takes half of the height and width
    """

    def __init__(self):
        self._values = []

    def calc(self, frame):
        frame_gray = skimage.color.rgb2gray(frame).astype(np.float32)

        # check half of input resolution
        width_hd, height_hd = frame_gray.shape[1] // 2, frame_gray.shape[0] // 2
        frame_gray_hd = cv2.resize(
            frame_gray, dsize=(width_hd, height_hd), interpolation=cv2.INTER_CUBIC
        )
        frame_gray_hd = cv2.resize(
            frame_gray_hd,
            dsize=(frame_gray.shape[1], frame_gray.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        v = float(skvideo.measure.psnr(frame_gray, frame_gray_hd)[0])
        if np.isinf(v):
            v = 1000
        self._values.append(v)
        return v


class Blockiness(Feature):
    """
    calculate blockness of a video, assume that compression blocks have the same NxN size, where N = [8, 16, 32, 64, 128],

    calculation performs the following steps, explained for N=8:

    - apply canny edge detection, K=[X,Y]-axis, normalized by num-rows/cols
    - calculate mean for all K summed values (A)
    - calculate for a shift (0,7) and for every 8th value of the K summed values the mean (B)
      - all shifts will be considered, and only the one with max_mean value is used
    - assume that the distribution should differe, if blocks are there
    - difference of A, and B is then the feature value

    overall blockiness value -- per frame:

    - value = :math:`\sqrt{(|x\_mean\_diff \cdot y\_mean\_diff | / 2^{(|max\_shift_x - max\_shift_y|/N)}}`

    """

    _blocksizes = [8, 16, 32, 64, 128]

    def __estimate_shift(self, values, blocksize):
        max_i = 0
        max_i_mean = values[max_i::blocksize].mean()
        for i in range(blocksize):
            curr_i_mean = values[i::blocksize].mean()
            if max_i_mean < curr_i_mean:
                max_i_mean = curr_i_mean
                max_i = i
        return max_i

    def calc(self, frame):
        frame_c = cv2.Canny(np.uint8(frame), 100, 200)
        xsums = frame_c.sum(axis=0) / (frame.shape[0])
        ysums = frame_c.sum(axis=1) / (frame.shape[1])
        xaxis_all = xsums.mean()
        yaxis_all = ysums.mean()

        blockiness_values = []
        for blocksize in self._blocksizes:
            max_i_x = self.__estimate_shift(xsums, blocksize)
            max_i_y = self.__estimate_shift(ysums, blocksize)

            xaxis_bsize = xsums[max_i_x::blocksize].mean()
            yaxis_bsize = ysums[max_i_y::blocksize].mean()
            v = {
                "x_mean_diff": xaxis_all - xaxis_bsize,
                "y_mean_diff": yaxis_all - yaxis_bsize,
                "max_i_x": max_i_x,
                "max_i_y": max_i_y,
            }
            v["diff"] = (
                np.sqrt(np.abs(v["x_mean_diff"] * v["y_mean_diff"]))
            ) / np.power(2, np.abs(max_i_x - max_i_y) / blocksize)
            blockiness_values.append(float(v["diff"]))
        r = max(blockiness_values)
        self._values.append(r)
        return r


class ImageFeature(Feature):
    """
    a generic image feature class,
    ususally all methods implemented in quat.visual.images
    can be passes as argument in the constructor
    """

    def __init__(self, img_f):
        """
        img_f needs to be a function that handles one frame
        """
        self._values = []
        self.img_f = img_f

    def calc(self, frame):
        v = self.img_f(frame)
        self._values.append(v)
        return v
