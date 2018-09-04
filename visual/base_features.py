#!/usr/bin/env python3

import cv2
import os
import json
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
    def __init__(self):
        self._values = []

    def calc(self, frame):
        pass

    def get_values(self):
        return self._values

    def _feature_filename(self, folder, video, name):
        bn = os.path.basename(os.path.splitext(video)[0])
        if name == "":
            name = self.__class__.__name__
        rfn = os.path.join(folder, bn + "_" + name + ".json")
        return rfn

    def load(self, folder, video, name=""):
        os.makedirs(folder, exist_ok=True)
        fn = self._feature_filename(folder, video, name)
        if os.path.isfile(fn):
            with open(fn) as ffp:
                j = json.load(ffp)
                self._values = j["values"]
                return j["values"]
        return False

    def store(self, folder, video, name=""):
        fn = self._feature_filename(folder, video, name)
        v = {
            "name": name,
            "values": self._values,
            "video": video
        }
        with open(fn, "w") as ffp:
            json.dump(v, ffp, indent=4, sort_keys=True)
        return fn

class MovementFeatures(Feature):
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
        #final = cv2.bilateralFilter(fgmask,9,75,75)
        if debug:
            cv2.imshow('o',cv2.resize(frame, (600,int(600 * file_height/file_width))))
            cv2.imshow('fg',cv2.resize(fgmask, (600,int(600 * file_height/file_width))))
            k = cv2.waitKey(30) & 0xff

        non_zero = cv2.countNonZero(fgmask)
        moving_percentage = (non_zero * 100) / (file_height * file_width)
        value = float(moving_percentage)
        self._values.append(value)
        return value

    def get_values(self):
        self._values[0] = 0
        return self._values


class CutDetectionFeatures(Feature):
    def __init__(self):
        self._values = []
        self._last_frame = None
        self._previous_diff_weighted = 0

    def calc(self, frame):
        # TODO change to skiimage
        # Scaling down the image speeds up computations and reduces false positives
        image = cv2.resize(frame, dsize=(320, 240),
                           interpolation=cv2.INTER_LANCZOS4)
        cut = 0
        # Converting the image type to a 16 bits signed integers image prevents over/under flows
        image = np.int16(image)

        if self._last_frame is not None:
            current_diff = np.std((image - self._last_frame).ravel())

            if current_diff > 30 and current_diff > 4 * self._previous_diff_weighted:
                # we detect this as a cut
                cut = 1

            self._previous_diff_weighted = self._previous_diff_weighted * 0.5 + 0.5 * current_diff
        self._last_frame = image
        self._values.append(cut)
        return cut



class SiFeatures(Feature):
    def __init__(self):
        self._values = []

    def calc(self, frame):
        def calculate_si(frame_data, magnitude=False):
            from scipy import ndimage
            if not magnitude:
                # P.910 description:
                return ndimage.sobel(frame_data).std()
            # Other implementation based on magnitude:
            dx = ndimage.sobel(frame_data, 1)  # horizontal derivative
            dy = ndimage.sobel(frame_data, 0)  # vertical derivative
            mag = np.hypot(dx, dy)  # magnitude
            mag = np.array(mag, dtype=np.uint8)
            return mag.std()
        value = calculate_si(frame)
        self._values.append(value)
        return value



class TiFeatures(Feature):
    def __init__(self):
        self._values = []
        self._previous_frame = None

    def calc(self, frame):
        def calculate_ti(frame_data, previous_frame_data):
            if previous_frame_data is None:
                return 0
            return (frame_data - previous_frame_data).std()
        value = calculate_ti(frame, self._previous_frame)
        self._previous_frame = frame
        self._values.append(value)
        return value



class TemporalFeatures(Feature):
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
                    "temporal.std": 0
                }
            spatial, temporal = strred_extract_info(previous_frame_data, frame_data)
            return {
                "spatial.mean": spatial.mean(),
                "spatial.std": spatial.std(),
                "temporal.mean": temporal.mean(),
                "temporal.std": temporal.std()
            }
        frame = skimage.color.rgb2gray(frame).astype(np.float32)
        value = calculate(frame, self._previous_frame)
        self._previous_frame = frame
        self._values.append(value)
        return value



class BlockMotion(Feature):
    """
    calculates block motion of two following frames,
    block size is estimated by 5% of the height of the input frames, to be resolution independent
    """
    def __init__(self):
        self._values = []
        self._last_frame = None

    def calc(self, frame):
        per_frame_values = {
            "blkm.zeros": 0,
            "blkm.ones": 0,
            "blkm.minusones": 0
        }
        if self._last_frame is not None:
            videodata = np.array([self._last_frame, frame])
            blocksize = int(self._last_frame.shape[0]*0.05)
            motion = skvideo.motion.blockMotion(videodata, method="SE3SS", mbSize=blocksize)
            m = motion[0].flatten()
            blk_motion_zeros = np.count_nonzero(m == 0) / len(m)
            blk_motion_ones = np.count_nonzero(m == 1) / len(m)
            blk_motion_minusones = np.count_nonzero(m == -1) / len(m)
            per_frame_values = {
                "blkm.zeros": blk_motion_zeros,
                "blkm.ones": blk_motion_ones,
                "blkm.minusones": blk_motion_minusones
            }
            self._values.append(per_frame_values)
        self._last_frame = frame
        return per_frame_values



class CuboidRow(Feature):
    WINDOW = 60  # maximum number of frames in sliding window
    def __init__(self, row):
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
    WINDOW = 60  # maximum number of frames in sliding window
    def __init__(self, col):
        self._col = col
        self._cols = []
        self._values = []

    def calc(self, frame):
        frame_gray = skimage.color.rgb2gray(frame)
        if len(self._cols) >= self.WINDOW:
            self._cols = self._cols[1:]
        col_i = int(self._col * (frame.shape[1] - 1))
        tmp = frame_gray[:,col_i].copy()  # copy reduces memory !
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
    def __init__(self):
        self._values = []

    def calc(self, frame):
        frame_gray = skimage.color.rgb2gray(frame).astype(np.float32)

        # check half of input resolution
        width_hd, height_hd = frame_gray.shape[1] // 2, frame_gray.shape[0] // 2
        frame_gray_hd = cv2.resize(frame_gray, dsize=(width_hd , height_hd), interpolation=cv2.INTER_CUBIC)
        frame_gray_hd = cv2.resize(frame_gray_hd, dsize=(frame_gray.shape[1] , frame_gray.shape[0]), interpolation=cv2.INTER_CUBIC)
        v = float(skvideo.measure.psnr(frame_gray, frame_gray_hd)[0])
        if np.isinf(v):
            v = 1000
        self._values.append(v)
        return v


class ImageFeature(Feature):
    def __init__(self, img_f):
        self._values = []
        self.img_f = img_f

    def calc(self, frame):
        v = self.img_f(frame)
        self._values.append(v)
        return v




