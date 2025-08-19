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
        self._num_repeat = kwargs.get("num_repeat", 1)
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
            timestamps=self.timestamps,
            lmdb_txn=lmdb_txn,
            num_repeat=self._num_repeat if split == "train" else 1,
            event_h=kwargs["event_height"] if lmdb_txn is not None else kwargs["event_raw_height"],
            event_w=kwargs["event_width"] if lmdb_txn is not None else kwargs["event_raw_width"],
            **event_cfg.PARAMS,
        )

        # Stereo objdet dataset
        isLoadCOCOFormat = kwargs.get("isLoadCOCOFormat", False)
        objdet_module = getattr(objdet, "base")
        self.objdet_dataset = objdet_module.StereoObjDetDataset(
            root=os.path.join(root, self._PATH_DICT["objdet"]),
            imageHeight=kwargs["event_rectified_height"],
            imageWidth=kwargs["event_rectified_width"],
            isLoadCOCOFormat=isLoadCOCOFormat,
            num_repeat=self._num_repeat if split == "train" else 1,
            timestamps=self.timestamps,
            dataset_type=split,
            max_num_keypoints=kwargs.get("max_num_keypoints", None)
        )

        # Disparity Dataset
        disparity_module = getattr(disparity, "base")
        img_metadata = {
            # 'h': self.event_dataset.event_h,
            # 'w': self.event_dataset.event_w,
            "h": crop_height,  # Note: disparity generation is based on objdet after cropping or padding.
            "w": crop_width,
            "h_recti": kwargs["event_rectified_height"],
            "w_recti": kwargs["event_rectified_width"]
        }
        self.disparity_dataset = disparity_module.DisparityDataset(
            img_metadata=img_metadata,
            event_data_sequence_length=len(self.event_dataset),
            num_repeat=self._num_repeat if split == "train" else 1
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
            if kwargs.get("randomcrop", False):
                transformsList.append(
                    transforms.RandomCrop(
                        event_module=event_module,
                        objdet_module=objdet_module,
                        crop_height=crop_height,
                        crop_width=crop_width,
                        no_value=self.objdet_dataset.NO_VALUE
                    )
                )
            else:
                transformsList.append(
                    transforms.Padding(
                        img_height=crop_height,
                        img_width=crop_width,
                        event_module=event_module,
                        no_event_value=self.event_dataset.NO_VALUE,
                        objdet_module=objdet_module,
                        no_objdet_value=self.objdet_dataset.NO_VALUE,
                        disparity_module=disparity_module,
                        no_disparity_value=self.disparity_dataset.NO_VALUE,
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
            transformsList = []
            if kwargs.get("randomcrop", False):
                transformsList.append(
                    transforms.RandomCrop(
                        event_module=event_module,
                        objdet_module=objdet_module,
                        crop_height=crop_height,
                        crop_width=crop_width,
                        no_value=self.objdet_dataset.NO_VALUE
                    )
                )
            else:
                transformsList.append(
                    transforms.Padding(
                        event_module=event_module,
                        img_height=crop_height,
                        img_width=crop_width,
                        no_event_value=self.event_dataset.NO_VALUE,
                        objdet_module=objdet_module,
                        no_objdet_value=self.objdet_dataset.NO_VALUE,
                        disparity_module=disparity_module,
                        no_disparity_value=self.disparity_dataset.NO_VALUE
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
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.event_dataset)

    def __getitem__(self, idx):
        data = self.load_data(idx)
        data = self.transforms(data)
        # print("baodebug: event timestamp: ", data['event']['timestamp'])
        # print("baodebug2: objdet timestamp: ", data['objdet']['timestamp'])
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
            "h_recti": self.event_dataset.event_h,
            "w_recti": self.event_dataset.event_w
        }
        if "EVENT_TENSOR_TYPE" in self.event_cfg and self.event_cfg.EVENT_TENSOR_TYPE == "secff":
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
        else:
            output["event"]["left"] = (
                output["event"]["left"].to(torch.float32)
            )
            output["event"]["right"] = (
                output["event"]["right"].to(torch.float32)
            )

        if 'disparity' in output or 'objdet' in output:
            output["gt_labels"] = {}
            if "disparity" in output:
                output["gt_labels"]["disparity"] = output['disparity']
            if "objdet" in output:
                output["gt_labels"]["objdet"] = output["objdet"]
        return output

    def load_data(self, idx):
        data = {}

        event_data = self.event_dataset[idx]
        objdet_data = self.objdet_dataset[idx]
        disparity_data = self.disparity_dataset[(idx, objdet_data)]

        data["file_index"] = idx
        data["end_timestamp"] = self.timestamps[idx // self._num_repeat]
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
