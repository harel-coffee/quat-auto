#!/usr/bin/env python3
import os
import json
import glob
import sys

import yaml
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/itu-p1203')

from itu_p1203.utils import mos_from_r, r_from_mos
from itu_p1203.p1203Pa import P1203Pa

from quat.video import advanced_pooling
from quat.utils.system import lglob


def _p_nams_av(audio, video):
    """
    use the pure p1201 integration module for audio and visual quality scores
    """
    if len(audio) + 1 == len(video):
        print("repreat last audio score")
        audio.append(audio[-1])

    if len(audio) != len(video):
        common_length = min(len(audio), len(video))
        print(f"audio and video length are not matching, truncate to min: audio:{len(audio)}, video:{len(video)}")
        audio = audio[0:common_length]
        video = video[0:common_length]
    audio = np.array(audio)
    video = np.array(video)
    _r_from_mos = np.vectorize(r_from_mos)
    _mos_from_r = np.vectorize(mos_from_r)

    Qcod_audio = 100 - _r_from_mos(audio)
    Qcod_video = 100 - _r_from_mos(video)
    av_coeffs = (100.8670, -0.3590, -0.9210, 0.00135)

    Q_audiovisual =  av_coeffs[0] + av_coeffs[1] * Qcod_audio + av_coeffs[2] * Qcod_video + av_coeffs[3] * Qcod_audio * Qcod_video
    MOS_audiovisual = _mos_from_r(Q_audiovisual)

    return MOS_audiovisual





class PNATSLongDB:
    def __init__(self, database_file):
        if not PNATSVideo.isPnatsDBFile(database_file):
            print("not a pnats db")
            raise Exception()
        self._restricted_yaml = False
        if "restricted" in database_file:
            print("restricted yaml")
            self._restricted_yaml = True

        yaml_file = lglob(database_file)[0]
        with open(yaml_file) as ry:
            y = yaml.load(ry)
        #print(y)
        if not "P2L" in database_file:
            print("database is not long, exit")
            #return
        self._db_folder = os.path.dirname(database_file)
        self._y = y

    def db_name(self):
        return os.path.basename(self._db_folder)

    def get_type(self):
        """ return 1 for mobile and 0 for pc database
        """
        if self._restricted_yaml:
            return 1 if self._y["type"] == "mobile" else 0
        # default case
        return 0

    def get_pvs_ids(self):
        if "pvsList" in self._y:
            return self._y["pvsList"]
        return [x for x in self._y["pvsInfo"]]

    def get_segment_files(self, pvsid):
        qc = lglob(self._db_folder + f"""/qualityChangeEventFiles/{pvsid}.qchanges""")[0]
        segment_filenames = []
        for j, s in pd.read_csv(qc).iterrows():
            segment_filenames.append(self._db_folder + "/videoSegments/" + s["segment_filename"])
        return segment_filenames

    def _calculate_audio_mos(self, audio_duration, audio_sample_rate, audio_codec, audio_bitrate):
        if audio_codec not in P1203Pa.VALID_CODECS:
            audio_codec = "ac3"
        return P1203Pa.audio_model_function(audio_codec, audio_bitrate)

    def get_audio_scores(self, pvsid):
        qc = lglob(self._db_folder + f"""/qualityChangeEventFiles/{pvsid}.qchanges""")[0]
        audio_scores = []
        for j, s in pd.read_csv(qc).iterrows():
            segment_file = self._db_folder + "/videoSegments/" + s["segment_filename"]
            try:
                audio_score = self._calculate_audio_mos(s["audio_duration"], s["audio_sample_rate"], s["audio_codec"], s["audio_bitrate"])
            except:
                audio_score = 5.0
            audio_scores.append({"segment_filename": segment_file, "audio_score": audio_score})
        return audio_scores

    def get_segment_durations(self, pvsid):
        qc = lglob(self._db_folder + f"""/qualityChangeEventFiles/{pvsid}.qchanges""")[0]
        durations = []
        for j, s in pd.read_csv(qc).iterrows():
            segment_file = self._db_folder + "/videoSegments/" + s["segment_filename"]
            durations.append(s["video_duration"])
        return durations

    def get_stalling(self, pvsid):
        buffer_events = []
        bf = lglob(self._db_folder + f"""/buffEventFiles/{pvsid}.buff""")
        if len(bf) == 0:
            return buffer_events
        bf = bf[0]
        with open(bf) as bf_fp:
            for l in bf_fp:
                buffer_events.append(json.loads(l))
        return buffer_events

    def get_mode_0(self, pvsid):
        qc = lglob(self._db_folder + f"""/qualityChangeEventFiles/{pvsid}.qchanges""")[0]
        df = pd.read_csv(qc)
        df["segment_filename"] = df["segment_filename"].apply(lambda x: self._db_folder + "/videoSegments/" + x)

        def unify_codec(x):
            if "h264" in x:
                return 0
            if "hevc" in x:
                return 1
            if "vp9" in x:
                return 2

        mode0_feature_values = []
        for r, i in df.iterrows():
            meta = {
                "viewing_distance": self._y.get("viewingDistance", 1.5),
                "type": 0 if self._y["type"].lower() == "pc" else 1,
                "display_height": self._y.get("displayHeight", 2160),
                "display_width": self._y.get("displayWidth", 3840),
            }

            meta["avg_frame_rate"] = i["video_frame_rate"]
            meta["bitrate"] = i["video_bitrate"]
            meta["codec"] = i["video_codec"]
            meta["height"] = i["video_height"]
            meta["width"] = i["video_width"]

            # mode0 base data
            mode0_features = {  # numbers are important here
                "framerate": float(meta["avg_frame_rate"]),
                "bitrate": float(meta["bitrate"]), # meta bitrate is in kbit/s
                "codec": unify_codec(meta["codec"]),
                "viewing_distance": meta["viewing_distance"],
                "resolution": int(meta["height"]) * int(meta["width"]),
                "type": meta["type"]
            }
            # mode0 extended features
            mode0_features["bpp"] = 1024 * mode0_features["bitrate"] / (mode0_features["framerate"] * mode0_features["resolution"])
            mode0_features["bitrate_log"] = np.log(mode0_features["bitrate"])
            mode0_features["framerate_norm"] = mode0_features["framerate"] / 60.0
            mode0_features["resolution_norm"] = mode0_features["resolution"] / (meta["display_height"] * meta["display_width"])
            mode0_feature_values.append(mode0_features)
        df["mode0"] = mode0_feature_values
        return df[["segment_filename", "mode0"]]

    def get_mode_1(self, pvsid):
        vfi = lglob(self._db_folder + f"/videoFrameInformation/{pvsid}.vfi")[0]
        df = pd.read_csv(vfi)
        segments = []
        mode1_feature_values = []
        for r, i in df.groupby(by="segment"):
            mode1_features = {}
            dk = i.copy()
            dk["pict_type"] = dk["frame_type"]
            dk["pkt_size"] = dk["size"]
            dk["pkt_size"] = dk["pkt_size"].apply(int)

            mode1_features = advanced_pooling((dk["pkt_size"] / dk["pkt_size"].max()).values, name="mode_1_framesizes")
            mode1_features["mode_1_iframe_ratio"] = dk["pict_type"][dk["pict_type"] == "I"].count() / len(dk)
            mode1_feature_values.append(mode1_features)
            segments.append(self._db_folder + "/videoSegments/" + r)

        du = pd.DataFrame({"segment_filenames": segments, "mode1": mode1_feature_values})
        return du

    def get_avpvs(self, pvsid):
        p = self._db_folder + f"""/avpvs/{pvsid}.*"""
        avpvs = lglob(p)
        if len(avpvs) >= 0:
            return avpvs[0]
        return None

    def get_src_vid(self, pvsid):
        if "pvsInfo" in self._y:
            return self._db_folder + "/srcVid/" + self._y["pvsInfo"][pvsid]["srcFile"]
        # only restricted yaml files are allowed here
        return None

    def get_score(self, pvsid, normMos=True):
        mos_file = lglob(self._db_folder + "/MOS_*_PC.csv")
        if len(mos_file) == 0:
            mos_file = lglob(self._db_folder + "/MOS_*_MO.csv")
        if len(mos_file) == 0:
            print("no mos file there")
            return
        mos_file = mos_file[0]
        df = pd.read_csv(mos_file)[["PVS_ID", "MOS"]]
        if os.path.isfile(self._db_folder + "/corrected_mos.json") and normMos:
            print("use corrected_mos values")
            with open(self._db_folder + "/corrected_mos.json") as cmos:
                cmos_values = pd.DataFrame(json.load(cmos)["values"])
                # print(cmos_values.head())
                cmos_values["MOS"] = cmos_values["NORM_MOS"]
                df = cmos_values[["PVS_ID", "MOS"]]

        return df[df["PVS_ID"] == pvsid]["MOS"].values[0]


class PNATSVideo:
    def isPnatsDBFile(database_file):
        return ".yaml" in database_file and "P2" in database_file

    def readSegmentFilesAndRatings(database_file, normMos=True, longDB=False):
        db_folder = os.path.dirname(database_file)
        mos_file = lglob(db_folder + "/MOS_*_PC.csv")
        if len(mos_file) == 0:
            mos_file = lglob(db_folder + "/MOS_*_MO.csv")
        mos_file = mos_file[0]

        df = pd.read_csv(mos_file)[["PVS_ID", "MOS"]]
        if os.path.isfile(db_folder + "/corrected_mos.json") and normMos:
            print("use corrected_mos values")
            with open(db_folder + "/corrected_mos.json") as cmos:
                cmos_values = pd.DataFrame(json.load(cmos)["values"])
                print(cmos_values.head())
                cmos_values["MOS"] = cmos_values["NORM_MOS"]
                df = cmos_values[["PVS_ID", "MOS"]]

        segment_filenames = []
        for i, x in df.iterrows():
            qc = lglob(db_folder + f"""/qualityChangeEventFiles/{x["PVS_ID"]}.qchanges""")[0]
            segment_filename = ""
            if not longDB:
                for j, s in pd.read_csv(qc).iterrows():
                    # there is only one segment in short databases
                    segment_filename = db_folder + "/videoSegments/" + s["segment_filename"]
                    break
            else:
                segment_filename = []
                for j, s in pd.read_csv(qc).iterrows():
                    segment_filename.append(db_folder + "/videoSegments/" + s["segment_filename"])
            segment_filenames.append(segment_filename)
        df["video"] = segment_filenames
        return df

    def isPnatsVideo(video):
        dbpath = os.path.dirname(video).replace("videoSegments", "").replace("avpvs", "")
        yaml_file = lglob(dbpath + "/*_restricted.yaml")[0]
        return os.path.isfile(yaml_file)

    def __init__(self, video):
        self._video = video
        v = os.path.basename(video)
        tmp = v.split("_")
        db = tmp[0]
        src_id = tmp[1]
        # FIXME: only valid for short databases
        hrc = tmp[2].replace("Q", "HRC") if "Q" in tmp[2] else tmp[2]
        dbpath = os.path.dirname(video).replace("videoSegments", "").replace("avpvs", "")
        yaml_file = lglob(dbpath + "/*_restricted.yaml")[0]
        with open(yaml_file) as ry:
            y = yaml.load(ry)

        self._dbpath = dbpath
        self._y = y
        self._db = db
        self._src_id = src_id
        self._hrc = hrc

    def get_mode0(self):
        qc = lglob(self._dbpath + f"/qualityChangeEventFiles/{self._db}_{self._src_id}_{self._hrc}.qchanges")[0]
        return pd.read_csv(qc).iloc[0]  # only valid for SHORT DB

    def get_mode0_ffprobe(self):
        """
        return a ffprobe compatibe mode0 meta data report
        """
        m0 = self.get_mode0()
        meta = {
            "avg_frame_rate": m0["video_frame_rate"],
            "bitrate": 1024 * m0["video_bitrate"], #.. in bit/s
            "codec": m0["video_codec"], #h264, hevc, vp9
            "height": m0["video_height"],
            "width": m0["video_width"],
            "type": 0 if self._y["type"].lower() == "pc" else 1,
            "viewing_distance": self._y.get("viewingDistance", 1.5),
            "display_height": self._y.get("displayHeight", 2160),
            "display_width": self._y.get("displayWidth", 3840),
        }
        return meta

    def get_mode1(self):
        vfi = lglob(self._dbpath + f"/videoFrameInformation/{self._db}_{self._src_id}_{self._hrc}.vfi")[0]
        return pd.read_csv(vfi)

    def get_mode1_ffprobe(self):
        """
        return a ffprobe compatibe mode1 meta data report
        """
        per_frame_infos = []
        for _, r in self.get_mode1().iterrows():
            p = {
                "pict_type": r["frame_type"],
                "pkt_size": r["size"]
            }
            per_frame_infos.append(p)

        mode1 = {
            "frames": per_frame_infos
        }
        return mode1
