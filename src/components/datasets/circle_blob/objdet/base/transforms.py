import torch
import numpy as np


class ToTensor:
    def __call__(self, sample):
        sample["bboxes"] = torch.from_numpy(sample["bboxes"])
        sample["labels"] = torch.from_numpy(sample["labels"])
        return sample


def ChangeBboxFormatToCenterBased(
    x_tl,
    y_tl,
    x_br,
    y_br,
    indexInstance
):
    """
    Returns:
        x_c,
        y_c,
        w_l,
        h_l,
        indexInstance
    """
    x_c = (x_tl + x_br) / 2
    y_c = (y_tl + y_br) / 2
    w_l = x_br - x_tl
    h_l = y_br - y_tl
    return np.concatenate(
        [
            column.reshape(-1, 1)
            for column in [
                x_c,
                y_c,
                w_l,
                h_l,
                indexInstance
            ]
        ],
        axis=1,
    )


def ChangeBboxFormatToCornerBased(
    x_c,
    y_c,
    w_l,
    h_l,
    indexInstance
):
    x_tl = x_c - w_l / 2
    y_tl = y_c - h_l / 2
    x_br = x_c + w_l / 2
    y_br = y_c + h_l / 2
    return np.concatenate(
        [
            column.reshape(-1, 1)
            for column in [
                x_tl,
                y_tl,
                x_br,
                y_br,
                indexInstance
            ]
        ],
        axis=1,
    )


class VerticalFlip:
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def __call__(self, sample):
        bboxes = np.copy(sample["bboxes"])
        bboxes_cformat = ChangeBboxFormatToCenterBased(
            *[bboxes[:, indexColumn] for indexColumn in range(5)]
        )
        y_c_new = self.img_height - bboxes_cformat[:, 1]
        sample["bboxes"] = ChangeBboxFormatToCornerBased(
            bboxes_cformat[:, 0],
            y_c_new,
            bboxes_cformat[:, 2],
            bboxes_cformat[:, 3],
            bboxes_cformat[:, 4]
        )
        return sample


class HorizontalFlip:
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def __call__(self, sample):
        """
        bboxes, labels
        """
        bboxes = np.copy(sample["bboxes"])
        bboxes_cformat = ChangeBboxFormatToCenterBased(
            *[bboxes[:, indexColumn] for indexColumn in range(5)]
        )
        x_c_new = self.img_width - bboxes_cformat[:, 0]
        sample["bboxes"] = ChangeBboxFormatToCornerBased(
            *[
                x_c_new,
                bboxes_cformat[:, 1],
                bboxes_cformat[:, 2],
                bboxes_cformat[:, 3],
                bboxes_cformat[:, 4]
            ]
        )
        return sample


class Crop:
    def __init__(self):
        pass

    def __call__(self, sample, offset_x, offset_y):
        start_y = offset_y
        start_x = offset_x

        sample["bboxes"][:, 0] -= start_x
        sample["bboxes"][:, 1] -= start_y
        sample["bboxes"][:, 2] -= start_x
        sample["bboxes"][:, 3] -= start_y

        return sample


class Resize:
    def __init__(self, downsample_ratio):
        self.downsample_ratio = downsample_ratio
    
    def __call__(self, sample):
        sample["bboxes"][:, 0] /= self.downsample_ratio
        sample["bboxes"][:, 1] /= self.downsample_ratio
        sample["bboxes"][:, 2] /= self.downsample_ratio
        sample["bboxes"][:, 3] /= self.downsample_ratio
        return sample
