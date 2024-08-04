import os
import csv
import copy

import numpy as np

import torch.utils.data

from . import disparity
from . import event
from . import transforms
from . import objdet


class SequenceDataset(torch.utils.data.Dataset):
    timestamps = None
    event_dataset = None
    objdet_dataset = None
    disparity_dataset = None
    _PATH_DICT = {
        "event": "events",  # Note: events are from lmdb now.
        "objdet": "objdet",
        # 'disparity': 'disparity',  # Note: use stereobbox from objdet to construct simple disparity labeling.
        "timestamps": "timestamps",
    }

    def __init__(
        self,
        root,
        split,
        sampling_ratio,
        event_cfg,
        crop_height,
        crop_width,
        num_workers=0,
        lmdb_txn=None,
        **kwargs,
    ):
        self.root = root
        self.split = split
        self.sampling_ratio = sampling_ratio
        self.event_cfg = event_cfg
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_workers = num_workers

        self.sequence_name = root.split("/")[-1]

        # Timestamps
        if split in ["train", "valid", "test"]:   
            if split == "test":
                self._PATH_DICT["timestamps"] = "timestamps_slam.txt"
            else:
                self._PATH_DICT["timestamps"] = "timestamps_objdet.txt"
            self.timestamps = np.loadtxt(
                os.path.join(root, self._PATH_DICT["timestamps"]), dtype="int64"
            )
            self.timestamp_to_index = {
                timestamp: idx for idx, timestamp in enumerate(self.timestamps)
            }
        else:
            raise NotImplementedError

        # Event Dataset
        event_module = getattr(event, event_cfg.NAME)
        event_root = os.path.join(root, self._PATH_DICT["event"])        
        self.event_dataset = event_module.EventDataset(
            root=event_root,
            sequence_name=self.sequence_name,
            sequence_length=len(self.timestamps),
            lmdb_txn=lmdb_txn,
            **event_cfg.PARAMS,
        )

        # Stereo objdet dataset
        objdet_module = getattr(objdet, "base")
        self.objdet_dataset = objdet_module.StereoObjDetDataset(
            root=os.path.join(root, self._PATH_DICT["objdet"]),
            img_height=crop_height,
            img_width=crop_width
        )

        # Disparity Dataset
        disparity_module = getattr(disparity, "base")
        img_metadata = {
            # 'h': self.event_dataset.event_h,
            # 'w': self.event_dataset.event_w,
            "h": crop_height,
            "w": crop_width
        }
        self.disparity_dataset = disparity_module.DisparityDataset(
            img_metadata=img_metadata
        )

        # self.timestamps = self.timestamps[[idx for idx in range(0, len(self.timestamps), sampling_ratio)]]  # Bug: timestamp_to_index will be wrong.

        # Transforms
        print("split: {}".format(split))
        if split in ["train", "trainval"]:
            transformsList = []
            if kwargs.get("randomhorizontalflip", False):
                transformsList.append(
                    transforms.RandomHorizontalFlip(
                        event_module=event_module,
                        disparity_module=disparity_module,
                        objdet_module=objdet_module,
                        img_height=crop_height,
                        img_width=crop_width,
                    )
                )
            transformsList.append(
                transforms.Padding(
                    event_module=event_module,
                    img_height=crop_height,
                    img_width=crop_width,
                    no_event_value=self.event_dataset.NO_VALUE,
                    no_disparity_value=self.disparity_dataset.NO_VALUE,
                    disparity_module=disparity_module,
                )
            )
            transformsList.append(
                transforms.ToTensor(
                    event_module=event_module,
                    disparity_module=disparity_module,
                    objdet_module=objdet_module,
                )
            )
            self.transforms = transforms.Compose(transformsList)
        elif split in ["valid", "test"]:
            self.transforms = transforms.Compose(
                [
                    transforms.Padding(
                        event_module=event_module,
                        img_height=crop_height,
                        img_width=crop_width,
                        no_event_value=self.event_dataset.NO_VALUE,
                        no_disparity_value=self.disparity_dataset.NO_VALUE,
                        disparity_module=disparity_module,
                    ),
                    transforms.ToTensor(
                        event_module=event_module,
                        disparity_module=disparity_module,
                        objdet_module=objdet_module,
                    ),
                ]
            )
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        data = self.load_data(idx)
        data = self.transforms(data)
        return data

    def collate_fn(self, batch):
        output = {}
        # Event
        domain = "event"
        if domain in batch[0].keys():
            output[domain] = self.event_dataset.collate_fn(
                [sample[domain] for sample in batch]
            )

        # objdet
        domain = "objdet"
        if domain in batch[0].keys():
            output[domain] = self.objdet_dataset.collate_fn(
                [oneInstance[domain] for oneInstance in batch]
            )

        # Others
        for key in batch[0].keys():
            if key not in ["event", "objdet"]:
                output[key] = torch.utils.data._utils.collate.default_collate(
                    [sample[key] for sample in batch]
                )

        output["image_metadata"] = {
            "h": self.crop_height,
            "w": self.crop_width,
            "h_cam": self.event_dataset.event_h,
            "w_cam": self.event_dataset.event_w
        }
        output["event"]["left"] = (
            output["event"]["left"]
            .permute(0, 2, 3, 1)
            .unsqueeze(1)
            .unsqueeze(4)
            .to(torch.float32)
        )
        output["event"]["right"] = (
            output["event"]["right"]
            .permute(0, 2, 3, 1)
            .unsqueeze(1)
            .unsqueeze(4)
            .to(torch.float32)
        )
        if 'disparity' in output and 'objdet' in output:
            output["gt_labels"] = {
                "disparity": output['disparity'],
                "objdet": output["objdet"]
            }
        return output

    def load_data(self, idx):
        data = {}
        event_data = self.event_dataset[(idx, self.timestamps[idx])]
        objdet_data = self.objdet_dataset[self.timestamps[idx]]
        disparity_data = self.disparity_dataset[(idx, objdet_data)]

        data["file_index"] = idx
        data["end_timestamp"] = self.timestamps[idx]
        if event_data is not None:
            data["event"] = event_data
        if objdet_data is not None:
            data["objdet"] = objdet_data
        if disparity_data is not None:
            data["disparity"] = disparity_data

        return data


def read_csv(csv_file):
    timestamps = []
    timestamp_to_index = {}
    with open(csv_file) as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            assert row[0] not in timestamps
            if row[0].isnumeric():
                timestamps.append(int(row[0]))
                timestamp_to_index[int(row[0])] = int(row[1])

    return np.asarray(timestamps), timestamp_to_index
