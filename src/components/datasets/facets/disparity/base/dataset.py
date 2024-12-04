import os
from PIL import Image
import cv2

import numpy as np

import torch.utils.data


class DisparityDataset(torch.utils.data.Dataset):
    NO_VALUE = 0.0
    img_metadata = None  # {'h', 'w'}
    disparity_cache = None  # cache disparity in memory

    def __init__(self, img_metadata):
        self.img_metadata = img_metadata
        self.disparity_cache = dict()

    def __len__(self):
        return 0  # Note: length depends on the length of the event dataset.

    def __getitem__(self, x):
        """
        Args:
            ...
            objdet_data: a dict with 'bboxes' (Nx10 tensor) and 'labels' ((N,) tensor) keys.
        Returns:
            disparity: h*w image
        """
        idx, objdet_data = x
        if objdet_data is None:
            return
        if idx in self.disparity_cache.keys():
            disparity = self.disparity_cache[idx]
        else:
            disparity = make_disparity(objdet_data, self.img_metadata, self.NO_VALUE)
            self.disparity_cache[idx] = disparity
        # disparity /= 1000  # Note: normalize disparity as in event_stereo_matching.py.
        return disparity

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch


def load_timestamp(root):
    return np.loadtxt(root, dtype="int64")


def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]


def load_disparity(root):
    disparity = np.array(Image.open(root)).astype(np.float32)
    return disparity


def make_disparity(objdet_data, metadata, no_value):
    """
    method:
        For all the left image bboxes, create approximate disparity inside the
        rectangle of bounding box.
        For other areas, set to no_value.
    """
    disparity = np.full((metadata["h"], metadata["w"]), no_value, dtype="float32")
    for bbox in objdet_data.get("bboxes", []):
        oneInstanceMask = np.zeros_like(disparity, dtype="uint8")
        rectangle = np.array(
            [[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype="int"
        )
        cv2.fillPoly(oneInstanceMask, pts=[rectangle], color=(1,))
        disparityValue = (bbox[2] + bbox[0]) / 2 - (bbox[5] + bbox[4]) / 2
        disparity[oneInstanceMask.view("bool")] = disparityValue
    return disparity
