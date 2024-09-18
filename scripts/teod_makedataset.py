#!/usr/bin/env python3
import argparse
import logging
import os
import lmdb
import numpy
import shutil
import cv2
from tqdm import tqdm
import sys, os
import json


def ConfigureLogger():
    formatter = logging.Formatter(
        "%(asctime)s - %(process)d - %(levelname)s: %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("Logger configured.")
    return logger


log = ConfigureLogger()

file_path = os.path.realpath(__file__)
log.info("src package path: %s", os.path.join(os.path.dirname(file_path), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(file_path), "..", "src"))
from utils.config import get_cfg
from components import datasets


def CreatePath(path, isOverwrite=True):
    if isOverwrite and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


class LmdbWriter:  # lmdb is multi read single write
    def __init__(self, write_path, map_size=1099511627776, isDummyMode=False):
        self.write_path = write_path
        self.map_size = map_size
        self.isDummyMode = isDummyMode
        if not isDummyMode:
            self.env = lmdb.open(write_path, map_size)
            self.txn = self.env.begin(write=True)

    def write(self, key, dataunit):
        if not self.isDummyMode:
            self.txn.put(key=key, value=dataunit)

    def commitchange(self):
        # commit change before ram is full
        if not self.isDummyMode:
            self.txn.commit()

    def endwriting(self):
        if not self.isDummyMode:
            self.env.close()


def ConfigureArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="path to where train, valid folders are.",
    )
    parser.add_argument(
        "--dataset_type", type=str, required=True, help="train or valid."
    )
    parser.add_argument("--lmdb_dir", type=str, required=True)
    parser.add_argument(
        "--view4label_dir", type=str, required=True
    )  # Note: a convenient events representation that will be used for labeling
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--seq_idx",
        type=int,
        required=True,
        help="index of the sequence in the lmdb dataset."
    )
    parser.add_argument(
        "--seq_idx_toselect",
        type=int,
        required=True,
        help="index to select the seq. in the dataset constant list"
    )
    parser.add_argument("--calib_path", type=str, default=None)

    args = parser.parse_args()
    args.world_size = 1
    args.num_workers = 4
    return args


def ConvertEventsToImage(events):
    """
    convert sbt stack to a uint8 image for visualization.
    """
    events = events.astype("float")  # copy and convert to float
    events = events - events.min()
    events = events / events.max() * 255
    return events.astype(numpy.uint8)


def UndistortAndRectifyStereoEvents(leftEvents, rightEvents, calib_dict):
    """
    Args:
        leftEvents: (h, w, 10) shape.
        ...
    """
    left_min = int(leftEvents.min())  # Note: should always be -1 in se-cff datasets.
    leftEvents = (leftEvents  - leftEvents.min()).astype('uint8')
    right_min = int(rightEvents.min())
    rightEvents = (rightEvents - rightEvents.min()).astype('uint8')

    left_groups = [leftEvents[..., :3], leftEvents[..., 3:6], leftEvents[..., 6:9], leftEvents[..., 9]]
    right_groups = [rightEvents[..., :3], rightEvents[..., 3:6], rightEvents[..., 6:9], rightEvents[..., 9]]
    # left_groups = [leftone for leftone in leftEvents.transpose(2, 0, 1)]
    # right_groups = [rightone for rightone in rightEvents.transpose(2, 0, 1)]
    left_groups_results, right_groups_results = [], []
    for leftone, rightone in zip(left_groups, right_groups):
        leftone_undist = cv2.undistort(leftone, calib_dict['undistort']['cam0']['kk'], calib_dict['undistort']['cam0']['distCoeff'], None, calib_dict['undistort']['cam0']['kk'])
        rightone_undist = cv2.undistort(rightone, calib_dict['undistort']['cam1']['kk'], calib_dict['undistort']['cam1']['distCoeff'], None, calib_dict['undistort']['cam1']['kk'])
        leftone_rect = cv2.remap(
            leftone_undist,
            calib_dict['stereo_rectify']['left_stereo_map'][0],
            calib_dict['stereo_rectify']['left_stereo_map'][1],
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0
        )    
        rightone_rect = cv2.remap(
            rightone_undist,
            calib_dict['stereo_rectify']['right_stereo_map'][0],
            calib_dict['stereo_rectify']['right_stereo_map'][1],
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0
        )
        
        imgHeight, imgWidth = leftone_rect.shape[:2]
        margin = calib_dict['stereo_rectify']['margin']        
        leftone_rect = leftone_rect[margin:imgHeight - margin, margin:imgWidth - margin]
        rightone_rect = rightone_rect[margin:imgHeight - margin, margin:imgWidth - margin]
        if leftone_rect.ndim == 2:
            leftone_rect = leftone_rect[..., numpy.newaxis]
            rightone_rect = rightone_rect[..., numpy.newaxis]
        
        left_groups_results.append(leftone_rect)
        right_groups_results.append(rightone_rect)
    leftEvents_rect = numpy.concatenate(left_groups_results, axis=-1).astype('int8')
    rightEvents_rect = numpy.concatenate(right_groups_results, axis=-1).astype('int8')
    leftEvents_rect += left_min
    rightEvents_rect += right_min
    leftEvents_rect = numpy.clip(leftEvents_rect, -1, 1)
    rightEvents_rect = numpy.clip(rightEvents_rect, -1, 1)

    return leftEvents_rect, rightEvents_rect


def main(args):
    # get dataset config
    cfg = get_cfg(args.config_path)
    log.info("loaded dataset_config: %r", cfg)

    # create dataloader
    dataset_type = args.dataset_type
    get_data_loader = getattr(
        datasets,
        cfg.DATASET.TRAIN.NAME,  # train, valid, test should share the same dataset class
    ).get_dataloader
    batch_size = (
        cfg.DATALOADER.TRAIN.PARAMS.batch_size
    )

    if dataset_type == "train":
        dataset_cfg = cfg.DATASET.TRAIN
        dataloader_cfg = cfg.DATALOADER.TRAIN
    elif dataset_type == "valid":
        dataset_cfg = cfg.DATASET.VALID
        dataloader_cfg = cfg.DATALOADER.VALID
    elif dataset_type == "test":
        dataset_cfg = cfg.DATASET.TEST
        dataloader_cfg = cfg.DATALOADER.VALID

    data_loader = get_data_loader(
        args=args,
        dataset_cfg=dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        is_distributed=False,
        defineSeqIdx=args.seq_idx_toselect,
        isDisableLmdbRead=True
    )
    data_iter = iter(data_loader)

    stereo_calib_dict = None
    if args.calib_path is not None:
        with open(args.calib_path, "r") as calibFile:
            stereo_calib_dict = json.load(calibFile)
            stereo_calib_dict['stereo_rectify']['left_stereo_map'][0] = numpy.array(stereo_calib_dict['stereo_rectify']['left_stereo_map'][0], dtype='int16')
            stereo_calib_dict['stereo_rectify']['left_stereo_map'][1] = numpy.array(stereo_calib_dict['stereo_rectify']['left_stereo_map'][1], dtype='int16')
            stereo_calib_dict['stereo_rectify']['right_stereo_map'][0] = numpy.array(stereo_calib_dict['stereo_rectify']['right_stereo_map'][0], dtype='int16')
            stereo_calib_dict['stereo_rectify']['right_stereo_map'][1] = numpy.array(stereo_calib_dict['stereo_rectify']['right_stereo_map'][1], dtype='int16')
            stereo_calib_dict['undistort']['cam0']['kk'] = numpy.array(stereo_calib_dict['undistort']['cam0']['kk'])
            stereo_calib_dict['undistort']['cam0']['distCoeff'] = numpy.array(stereo_calib_dict['undistort']['cam0']['distCoeff'])
            stereo_calib_dict['undistort']['cam1']['kk'] = numpy.array(stereo_calib_dict['undistort']['cam1']['kk'])
            stereo_calib_dict['undistort']['cam1']['distCoeff'] = numpy.array(stereo_calib_dict['undistort']['cam1']['distCoeff'])

    # iterate over the dataset
    pbar = tqdm(total=len(data_loader.dataset) // batch_size)
    CreatePath(args.lmdb_dir, isOverwrite=False)
    CreatePath(os.path.join(args.view4label_dir, "{}".format(args.seq_idx), "{}_left".format(args.seq_idx)))
    CreatePath(os.path.join(args.view4label_dir, "{}".format(args.seq_idx), "{}_right".format(args.seq_idx)))
    lmdb_writer = LmdbWriter(args.lmdb_dir, isDummyMode=False)
    with open(
        os.path.join(args.lmdb_dir, "{}_timestamp.txt".format(args.seq_idx)), "w"
    ) as tsFile:
        indexSavedBatch = 0
        for indexBatch in range(len(data_loader.dataset) // batch_size):
            batch_data = next(data_iter)
            for indexInBatch in range(batch_size):
                ts = int(batch_data["end_timestamp"][indexInBatch].numpy())
                tsFile.write(str(ts) + "\n")
                
                print("seq_idx: {}, index frame: {}".format(args.seq_idx, indexSavedBatch * batch_size + indexInBatch))
                code_l = "%03d_%06d_l" % (
                    args.seq_idx,
                    indexSavedBatch * batch_size + indexInBatch,
                )
                code_l = code_l.encode()
                leftEvents = numpy.ascontiguousarray(
                    numpy.squeeze(batch_data["event"]["left"][indexInBatch].numpy())
                )                

                code_r = "%03d_%06d_r" % (
                    args.seq_idx,
                    indexSavedBatch * batch_size + indexInBatch,
                )
                code_r = code_r.encode()
                rightEvents = numpy.ascontiguousarray(
                    numpy.squeeze(batch_data["event"]["right"][indexInBatch].numpy())
                ).astype("int8")

                if stereo_calib_dict is None:
                    leftEvents = leftEvents.astype("int8")
                    rightEvents = rightEvents.astype("int8")
                else:
                    # hack, FIXME!!: in the unitree dataset, left and right camera are already corrected, but the calib file is not fixed.
                    # leftEvents, rightEvents = UndistortAndRectifyStereoEvents(leftEvents, rightEvents, stereo_calib_dict)
                    rightEvents, leftEvents = UndistortAndRectifyStereoEvents(rightEvents, leftEvents, stereo_calib_dict)

                lmdb_writer.write(code_l, leftEvents)
                lmdb_writer.write(code_r, rightEvents)

                code_ts = "%03d_%06d_ts" % (
                    args.seq_idx,
                    indexSavedBatch * batch_size + indexInBatch,
                )
                code_ts = code_ts.encode()
                lmdb_writer.write(code_ts, numpy.array([ts], dtype="int"))

                leftView = ConvertEventsToImage(leftEvents[..., 0])
                rightView = ConvertEventsToImage(rightEvents[..., 0])
                leftViewPath = os.path.join(
                    args.view4label_dir,
                    "{}".format(args.seq_idx),
                    "{}_left".format(args.seq_idx),
                    "{}.png".format(str(ts).zfill(12)),
                )
                rightViewPath = os.path.join(
                    args.view4label_dir,
                    "{}".format(args.seq_idx),
                    "{}_right".format(args.seq_idx),
                    "{}.png".format(str(ts).zfill(12)),
                )
                cv2.imwrite(leftViewPath, leftView)
                cv2.imwrite(rightViewPath, rightView)
            pbar.update(1)
            indexSavedBatch += 1
        log.info("commiting dataset...")
        lmdb_writer.commitchange()
        lmdb_writer.endwriting()
        log.info("done.")


if __name__ == "__main__":
    args = ConfigureArguments()
    main(args)
