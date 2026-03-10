import os
from PIL import Image
import albumentations as A
import cv2
import torch.utils.data
import numpy

from . import objdet, imagedata
from .transforms import make_mujinextended_transforms
from .transforms import ConvertCoco


def get_geometric_transform(height: int, width: int):
    return A.Compose([
            # 1. Shift and Scale ONLY (Rotation locked to 0)
            A.ShiftScaleRotate(
                shift_limit=0.1,    # Random translation
                scale_limit=0.2,    # Random scaling
                rotate_limit=0,     # NO ROTATION
                border_mode=cv2.BORDER_CONSTANT, 
                value=0,            # Padding value for the image
                p=0.5
            ),
            
            # 2. Random Vertical Flip (Safe for stereo)
            A.VerticalFlip(p=0.5),

            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(
                min_height=height, # Let the divisor handle the final size.
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            )
        ], 
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),  # cxcywh format boxes.
    )


def do_geometric_transform(transforms: A.Compose, data: dict):
    transformed = transforms(**data)
    return transformed


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
        **kwargs,
    ):
        self.root = root
        self.split = split
        self.transforms_geom = None
        if split != "test":
            self.transforms_geom = get_geometric_transform(kwargs["original_height"], kwargs["original_width"])
        self.transforms = make_mujinextended_transforms(
            split,
            kwargs["resolution"],
            multi_scale=kwargs["multi_scale"],
            expanded_scales=kwargs["expanded_scales"],
            skip_random_resize=not kwargs["do_random_resize_via_padding"],
            patch_size=kwargs["patch_size"],
            num_windows=kwargs["num_windows"],
        )
        self.prepare = ConvertCoco(include_masks=kwargs["include_masks"])
        self.sequence_name = root.split("/")[-1]

        # objdet dataset
        objdet_module = getattr(objdet, "base")
        self.objdet_dataset = objdet_module.ObjDetDataset(
            root=os.path.join(root, self._PATH_DICT["objdet"]),
            num_repeat=kwargs["num_repeat"])

        # image dataset
        imagedata_module = getattr(imagedata, "base")
        self.image_dataset = imagedata_module.ImageDataset(
            root=os.path.join(root, self._PATH_DICT["imagedata"]),
            num_repeat=kwargs["num_repeat"])

        self._no_transform: bool = False

    @property
    def no_transform(self):
        return self._no_transform

    @no_transform.setter
    def no_transform(self, value: bool):
        self._no_transform = value

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        data = self.load_data(idx)
        img, target = data["imagedata"], data["objdet"]
        if self.no_transform:
            return img, target
        target = {"image_id": target[0]["image_id"], "annotations": target}
        img, target = self.prepare(img, target)
        if self.transforms_geom and self.split == "train":
            transformed = do_geometric_transform(
                self.transforms_geom,
                {
                    "image": img,
                    "bboxes": target["boxes"].numpy(),
                    "labels": target["labels"].numpy(),
                },
            )
            img = transformed["image"]
            target["labels"] = torch.from_numpy(transformed["labels"])
            target["boxes"] = torch.from_numpy(transformed["bboxes"])
        img, target = self.transforms(img, target)
        return {"objdet": target, "imagedata": img, "data_index": data["data_index"]}

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

        output["gt_labels"] = {
            "objdet": output.pop("objdet"),
        }
        output["data_index"] = torch.tensor([sample["data_index"] for sample in batch])

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
