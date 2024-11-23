import torch
import numpy as np


class ToTensor:
    def __call__(self, sample):
        sample["bboxes"] = torch.from_numpy(sample["bboxes"])
        sample["labels"] = torch.from_numpy(sample["labels"])
        sample["keypt1_masks"] = torch.from_numpy(sample["keypt1_masks"])
        sample["keypt2_masks"] = torch.from_numpy(sample["keypt2_masks"])
        return sample


def ChangeBboxFormatToCenterBased(
    x_tl,
    y_tl,
    x_br,
    y_br,
    x_tl_r,
    x_br_r,
    delta_x_keypt1,
    delta_y_keypt1,
    delta_x_keypt2,
    delta_y_keypt2,
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
    x_keypt1 = delta_x_keypt1 * w_l + x_tl
    y_keypt1 = delta_y_keypt1 * h_l + y_tl
    x_keypt2 = delta_x_keypt2 * w_l + x_tl
    y_keypt2 = delta_y_keypt2 * h_l + y_tl
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
    delta_x_keypt1 = (x_keypt1 - x_tl) / w_l
    delta_y_keypt1 = (y_keypt1 - y_tl) / h_l
    delta_x_keypt2 = (x_keypt2 - x_tl) / w_l
    delta_y_keypt2 = (y_keypt2 - y_tl) / h_l
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
                delta_x_keypt1,
                delta_y_keypt1,
                delta_x_keypt2,
                delta_y_keypt2,
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
        return sample
