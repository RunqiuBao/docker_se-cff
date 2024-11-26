import torch
import numpy as np
import importlib

if importlib.metadata.version('torchvision') == '0.15.2':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Mask, Image, Video
    from torchvision.transforms.v2 import SanitizeBoundingBox as SanitizeBoundingBoxes
    _boxes_keys = ['format', 'spatial_size']

elif '0.17' > importlib.metadata.version('torchvision') >= '0.16':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

elif importlib.metadata.version('torchvision') >= '0.17':
    import torchvision
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

else:
    raise RuntimeError('Please make sure torchvision version >= 0.15.2')


class ToTensor:
    def __call__(self, sample):
        for key, value in sample.items():
            if isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(value)
            elif isinstance(sample[key], dict):
                for subkey in sample[key].keys():
                    sample[key][subkey] = torch.from_numpy(sample[key][subkey])
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


class Padding:
    def __init__(self, img_height, img_width, no_objdet_value):
        self.img_height = img_height
        self.img_width = img_width
        self.no_objdet_value = no_objdet_value

    def __call__(self, sample):
        if "pmap" in sample:
            ori_height, ori_width = sample["pmap"]["left"].shape[-2:]

            bottom_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert bottom_pad >= 0 and right_pad >= 0
            sample["pmap"]["left"] = np.lib.pad(
                sample["pmap"]["left"],
                ((0, bottom_pad), (0, right_pad)),
                mode="constant",
                constant_values=self.no_objdet_value,
            )
            sample["pmap"]["right"] = np.lib.pad(
                sample["pmap"]["right"],
                ((0, bottom_pad), (0, right_pad)),
                mode="constant",
                constant_values=self.no_objdet_value,
            )
        elif "left" in sample and "right" in sample:
            ori_height, ori_width = sample["left"]["segMaps"].shape[-2:]

            bottom_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width
            for side in ["left", "right"]:
                sample[side]["segMaps"] = np.lib.pad(
                    sample[side]["segMaps"],
                    ((0, 0), (0, bottom_pad), (0, right_pad)),
                    mode="constant",
                    constant_values=self.no_objdet_value,
                )
        else:
            raise NotImplementedError

        return sample


def convert_to_tv_tensor(tensor: torch.Tensor, key: str, box_format='xyxy', spatial_size=None) -> torch.Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in ('boxes', 'masks', ), "Only support 'boxes' and 'masks'"
    
    if key == 'boxes':
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == 'masks':
       return Mask(tensor)


class ConvertBboxes:
    def __init__(self, img_height: int, img_width: int, output_fmt: str="cxcywh", is_normalize: bool=True):
        self.img_height = img_height
        self.img_width = img_width
        self.output_fmt = output_fmt
        self.normalize = is_normalize

    def __call__(self, sample):
        assert "boxes" in sample.get("left", None) and isinstance(sample["left"]["boxes"], torch.Tensor)
        spatial_size = (self.img_height, self.img_width)
        for side in sample.keys():
            sample[side]["boxes"] = self._ConvertBboxFormat(sample[side]["boxes"], self.output_fmt, spatial_size, self.normalize)
        return sample

    @staticmethod
    def _ConvertBboxFormat(bboxes, output_fmt, spatial_size, normalize):
        in_fmt = "xywh"
        if in_fmt != output_fmt:
            bboxes = torchvision.ops.box_convert(bboxes, in_fmt=in_fmt, out_fmt=output_fmt.lower())

        bboxes = convert_to_tv_tensor(bboxes, key='boxes', box_format=output_fmt.upper(), spatial_size=spatial_size)

        if normalize:
            bboxes = bboxes / torch.tensor(spatial_size[::-1]).tile(2)[None]
        return bboxes.float()
