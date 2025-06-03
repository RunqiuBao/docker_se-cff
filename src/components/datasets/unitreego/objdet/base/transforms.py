import torch
import numpy as np


class ToTensor:
    def __call__(self, sample):
        sample["bboxes"] = torch.from_numpy(sample["bboxes"])
        sample["labels"] = torch.from_numpy(sample["labels"])
        sample["keypts"] = torch.from_numpy(sample["keypts"])
        return sample


def ChangeBboxFormatToCenterBased(
    x_tl,
    y_tl,
    x_br,
    y_br,
    x_tl_r,
    x_br_r,
    x_keypt1,
    y_keypt1,
    x_keypt2,
    y_keypt2,
):
    """
    Returns:
        x_c,
        y_c,
        w_l,
        h_l,
        x_c_r,
        w_r,
        x_keypt1,
        y_keypt1,
        x_keypt2,
        y_keypt2
    """
    x_c = (x_tl + x_br) / 2
    y_c = (y_tl + y_br) / 2
    w_l = x_br - x_tl
    h_l = y_br - y_tl
    x_c_r = (x_tl_r + x_br_r) / 2
    w_r = x_br_r - x_tl_r
    return np.concatenate(
        [
            column.reshape(-1, 1)
            for column in [
                x_c,
                y_c,
                w_l,
                h_l,
                x_c_r,
                w_r,
                x_keypt1,
                y_keypt1,
                x_keypt2,
                y_keypt2,
            ]
        ],
        axis=1,
    )


def ChangeBboxFormatToCornerBased(
    x_c, y_c, w_l, h_l, x_c_r, w_r, x_keypt1, y_keypt1, x_keypt2, y_keypt2
):
    x_tl = x_c - w_l / 2
    y_tl = y_c - h_l / 2
    x_br = x_c + w_l / 2
    y_br = y_c + h_l / 2
    x_tl_r = x_c_r - w_r / 2
    x_br_r = x_c_r + w_r / 2
    return np.concatenate(
        [
            column.reshape(-1, 1)
            for column in [
                x_tl,
                y_tl,
                x_br,
                y_br,
                x_tl_r,
                x_br_r,
                x_keypt1,
                y_keypt1,
                x_keypt2,
                y_keypt2,
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
            *[bboxes[:, indexColumn] for indexColumn in range(10)]
        )
        y_c_new = self.img_height - bboxes_cformat[:, 1]
        y_keypt1_new = self.img_height - bboxes_cformat[:, 7]
        y_keypt2_new = self.img_height - bboxes_cformat[:, 9]
        sample["bboxes"] = ChangeBboxFormatToCornerBased(
            bboxes_cformat[:, 0],
            y_c_new,
            bboxes_cformat[:, 2],
            bboxes_cformat[:, 3],
            bboxes_cformat[:, 4],
            bboxes_cformat[:, 5],
            bboxes_cformat[:, 6],
            y_keypt1_new,
            bboxes_cformat[:, 8],
            y_keypt2_new,
        )
        keypts = np.copy(sample["keypts"])
        keypts[:, :, 1] = self.img_height - keypts[:, :, 1]
        sample["keypts"] = keypts
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
            *[bboxes[:, indexColumn] for indexColumn in range(10)]
        )
        x_c_new = self.img_width - bboxes_cformat[:, 0]
        x_c_r_new = self.img_width - bboxes_cformat[:, 4]
        x_keypt1_new = self.img_width - bboxes_cformat[:, 6]
        x_keypt2_new = self.img_width - bboxes_cformat[:, 8]
        sample["bboxes"] = ChangeBboxFormatToCornerBased(
            *[
                x_c_new,
                bboxes_cformat[:, 1],
                bboxes_cformat[:, 2],
                bboxes_cformat[:, 3],
                x_c_r_new,
                bboxes_cformat[:, 5],
                x_keypt1_new,
                bboxes_cformat[:, 7],
                x_keypt2_new,
                bboxes_cformat[:, 9],
            ]
        )
        keypts = np.copy(sample["keypts"])
        keypts[:, :, 0] = self.img_width - keypts[:, :, 0]
        sample["keypts"] = keypts
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
        sample["bboxes"][:, 4] -= start_x
        sample["bboxes"][:, 5] -= start_x
        sample["bboxes"][:, 6] -= start_x
        sample["bboxes"][:, 7] -= start_y
        sample["bboxes"][:, 8] -= start_x
        sample["bboxes"][:, 9] -= start_y

        sample["keypts"][:, :, 0] -= start_x
        sample["keypts"][:, :, 1] -= start_y
        return sample
