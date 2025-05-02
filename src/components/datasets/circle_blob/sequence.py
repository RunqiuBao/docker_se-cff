import os
import csv
import copy

import numpy as np

import torch.utils.data

from . import transforms
from . import objdet, imagedata


class SequenceDataset(torch.utils.data.Dataset):
    objdet_dataset = None
    imagedata_dataset = None
    _PATH_DICT = {
        "objdet": "objdet",
        "imagedata": "imagedata",
        "datalist": "datalist.txt",
    }

    def __init__(
        self,
        root,
        split,
        crop_height,
        crop_width,
        num_workers=0,
        **kwargs,
    ):
        self.root = root
        self.split = split
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_workers = num_workers

        self.datalist = np.loadtxt(
            os.path.join(root, self._PATH_DICT["datalist"]), dtype="string"
        )

        # objdet dataset
        objdet_module = getattr(objdet, "base")
        self.objdet_dataset = objdet_module.ObjDetDataset(
            root=os.path.join(root, self._PATH_DICT["objdet"]),
            img_height=crop_height,
            img_width=crop_width
        )

        # image dataset
        imagedata_module = getattr(imagedata, "base")
        self.image_dataset = imagedata_module.ImageDataset(
            root=os.path.join(root, self._PATH_DICT["imagedata"]),
            img_height=crop_height,
            img_width=crop_width
        )

        # Transforms
        print("split: {}".format(split))
        if split in ["train", "trainval"]:
            transformsList = []
            if kwargs.get("randomhorizontalflip", False):
                transformsList.append(
                    transforms.RandomHorizontalFlip(
                        imagedata_module=imagedata_module,
                        objdet_module=objdet_module,
                        img_height=crop_height,
                        img_width=crop_width,
                    )
                )
            transformsList.append(
                transforms.Padding(
                    imagedata_module=imagedata_module,
                    img_height=crop_height,
                    img_width=crop_width,
                    no_value=self.image_dataset.NO_VALUE,
                )
            )
            transformsList.append(
                transforms.ToTensor(
                    imagedata_module=imagedata_module,
                    objdet_module=objdet_module
                )
            )
            self.transforms = transforms.Compose(transformsList)
        elif split in ["valid", "test"]:
            self.transforms = transforms.Compose(
                [
                    transforms.Padding(
                        imagedata_module=imagedata_module,
                        img_height=crop_height,
                        img_width=crop_width,
                        no_value=self.image_dataset.NO_VALUE,
                    ),
                    transforms.ToTensor(
                        imagedata_module=imagedata_module,
                        objdet_module=objdet_module
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
        # imagedata
        domain = "imagedata"
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
            if key not in ["imagedata", "objdet"]:
                output[key] = torch.utils.data._utils.collate.default_collate(
                    [sample[key] for sample in batch]
                )

        output["image_metadata"] = {
            "h": self.crop_height,
            "w": self.crop_width,
            "h_original": self.image_dataset.original_h,
            "w_original": self.image_dataset.original_w
        }
        if 'imagedata' in output and 'objdet' in output:
            output["gt_labels"] = {
                "imagedata": output['imagedata'],
                "objdet": output["objdet"]
            }
        return output

    def load_data(self, idx):
        data = {}
        image_data = self.image_dataset[(idx, self.timestamps[idx])]
        objdet_data = self.objdet_dataset[self.timestamps[idx]]

        data["file_index"] = idx
        data["end_timestamp"] = self.timestamps[idx]
        if objdet_data is not None:
            data["objdet"] = objdet_data
        if image_data is not None:
            data["imagedata"] = image_data

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
