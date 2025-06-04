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

        # objdet dataset
        objdet_module = getattr(objdet, "base")
        self.objdet_dataset = objdet_module.ObjDetDataset(
            root=os.path.join(root, self._PATH_DICT["objdet"]),
            num_repeat=5 if split == "train" else 1)

        # image dataset
        imagedata_module = getattr(imagedata, "base")
        self.image_dataset = imagedata_module.ImageDataset(
            root=os.path.join(root, self._PATH_DICT["imagedata"]),
            num_repeat=5 if split == "train" else 1)

        # Transforms
        print("split: {}".format(split))
        if split in ["train", "trainval"]:
            transformsList = []
            if kwargs.get("randomcropflip", False):
                transformsList.append(
                    transforms.RandomCrop(
                        imagedata_module=imagedata_module,
                        objdet_module=objdet_module,
                        crop_height=crop_height,
                        crop_width=crop_width,
                        no_value=self.image_dataset.NO_VALUE
                    )
                )
                transformsList.append(
                    transforms.RandomHorizontalFlip(
                        imagedata_module=imagedata_module,
                        objdet_module=objdet_module,
                        img_height=crop_height,
                        img_width=crop_width,
                    )
                )
                transformsList.append(
                    transforms.RandomVerticalFlip(
                        imagedata_module=imagedata_module,
                        objdet_module=objdet_module,
                        img_height=crop_height,
                        img_width=crop_width,
                    )
                )
                transformsList.append(
                    transforms.Resize(
                        imagedata_module=imagedata_module,
                        objdet_module=objdet_module,
                        downsample_ratio=kwargs["downsample_ratio"]
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
                        no_value=self.image_dataset.NO_VALUE
                    ),
                    transforms.Resize(
                        imagedata_module=imagedata_module,
                        objdet_module=objdet_module,
                        downsample_ratio=kwargs["downsample_ratio"]
                    ),
                    transforms.ToTensor(
                        imagedata_module=imagedata_module,
                        objdet_module=objdet_module
                    )
                ]
            )
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        data = self.load_data(idx)
        data = self.transforms(data)
        return data

    def collate_fn(self, batch):
        output = {}
        # imagedata
        domain = "imagedata"
        if domain in batch[0].keys():
            output[domain] = self.image_dataset.collate_fn(
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
                "objdet": output["objdet"]
            }
        return output

    def load_data(self, idx):
        data = {}
        image_data = self.image_dataset[idx]
        objdet_data = self.objdet_dataset[idx]

        data["data_index"] = idx
        if objdet_data is not None:
            data["objdet"] = objdet_data
        if image_data is not None:
            data["imagedata"] = image_data

        return data
