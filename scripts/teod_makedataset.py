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

def ConfigureLogger():
    formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s: %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("Logger configured.")
    return logger

log = ConfigureLogger()

file_path = os.path.realpath(__file__)
log.info('src package path: %s', os.path.join(os.path.dirname(file_path), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(file_path), '..', 'src'))
from utils.config import get_cfg
from components import datasets


def CreatePath(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


class LmdbWriter():  # lmdb is multi read single write
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
    parser.add_argument('--data_root', type=str, required=True, help="path to where train, valid folders are.")
    parser.add_argument('--dataset_type', type=str, required=True, help='train or valid.')
    parser.add_argument('--lmdb_dir', type=str, required=True)
    parser.add_argument('--view4label_dir', type=str, required=True)  # Note: a convenient events representation that will be used for labeling
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--seq_idx', type=int, required=True, help='index of this sequence in the dataset.')

    args = parser.parse_args()
    args.world_size = 1
    args.num_workers = 4
    return args


def ConvertEventsToImage(events):
    '''
    convert sbt stack to a uint8 image for visualization.
    '''
    events = events.astype('float')  # copy and convert to float
    events = events - events.min()
    events = events / events.max() * 255
    return events.astype(numpy.uint8)


def main(args):
    # get dataset config
    cfg = get_cfg(args.config_path)
    log.info("loaded dataset_config: %r", cfg)

    # create dataloader
    dataset_type = 'train' if args.dataset_type == 'train' else 'valid'
    get_data_loader = getattr(datasets, cfg.DATASET.TRAIN.NAME if dataset_type == 'train' else cfg.DATASET.VALID.NAME).get_dataloader
    data_loader = get_data_loader(
        args=args,
        dataset_cfg=cfg.DATASET.TRAIN if dataset_type == 'train' else cfg.DATASET.VALID,
        dataloader_cfg=cfg.DATALOADER.TRAIN if dataset_type == 'train' else cfg.DATALOADER.VALID,
        is_distributed=False,
        defineSeqIdx=args.seq_idx
    )
    data_iter = iter(data_loader)

    # iterate over the dataset
    pbar = tqdm(total=len(data_loader.dataset))
    CreatePath(args.lmdb_dir)
    CreatePath(os.path.join(args.view4label_dir, '{}_left'.format(args.seq_idx)))
    CreatePath(os.path.join(args.view4label_dir, '{}_right'.format(args.seq_idx)))
    lmdb_writer = LmdbWriter(args.lmdb_dir, isDummyMode=False)
    with open(os.path.join(args.lmdb_dir, '{}_timestamp.txt'.format(args.seq_idx)), 'w') as tsFile:
        for indexBatch in range(len(data_loader.dataset)):
            batch_data = next(data_iter)
            for indexInBatch in range(batch_data['event']['left'].shape[0]):
                ts = int(batch_data['end_timestamp'][indexInBatch].numpy())

                tsFile.write(str(ts) + '\n')

                code = '%03d_l' % (args.seq_idx)
                code = code.encode()
                leftEvents = numpy.ascontiguousarray(
                    numpy.squeeze(
                        batch_data['event']['left'][indexInBatch].numpy()
                    )
                ).astype('int8')
                lmdb_writer.write(code, leftEvents)

                code = '%03d_r' % (args.seq_idx)
                code = code.encode()
                rightEvents = numpy.ascontiguousarray(
                    numpy.squeeze(
                        batch_data['event']['right'][indexInBatch].numpy()
                    )
                ).astype('int8')
                lmdb_writer.write(code, rightEvents)

                code = '%03d_ts' % (args.seq_idx)
                code = code.encode()
                lmdb_writer.write(code, numpy.array([ts], dtype='int'))

                leftView = ConvertEventsToImage(leftEvents[..., -3])
                rightView = ConvertEventsToImage(rightEvents[..., -3])
                leftViewPath = os.path.join(args.view4label_dir, '{}_left'.format(args.seq_idx), '{}.png'.format(str(ts).zfill(8)))
                rightViewPath = os.path.join(args.view4label_dir, '{}_right'.format(args.seq_idx), '{}.png'.format(str(ts).zfill(8)))
                cv2.imwrite(leftViewPath, leftView)
                cv2.imwrite(rightViewPath, rightView)
            pbar.update(1)
        log.info("commiting dataset...")
        lmdb_writer.commitchange()
        lmdb_writer.endwriting()
        log.info("done.")


if __name__ == "__main__":
    args = ConfigureArguments()
    main(args)
