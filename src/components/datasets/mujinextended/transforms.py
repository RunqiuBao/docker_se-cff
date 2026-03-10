from typing import List, Any, Tuple, Optional, Dict, Union

import random
import numpy as np
import torch
import pycocotools.mask as coco_mask
from torchvision.transforms import v2 as T
# from detectron2.data import transforms as DT
from torchvision.transforms.v2 import functional as F
from components.models.rfdetr.util.box_ops import box_xyxy_to_cxcywh
from components.models.rfdetr.util.misc import interpolate


def convert_coco_poly_to_mask(segmentations: List[Any], height: int, width: int) -> torch.Tensor:
    """Convert polygon segmentation to a binary mask tensor of shape [N, H, W].
    Requires pycocotools.
    """
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


class ConvertCoco(object):

    def __init__(self, include_masks: bool = False) -> None:
        self.include_masks = include_masks

    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = image.shape[:2]

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # add segmentation masks if requested, otherwise ensure consistent key when include_masks=True
        if self.include_masks:
            if len(anno) > 0 and 'segmentation' in anno[0]:
                segmentations = [obj.get("segmentation", []) for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                if masks.numel() > 0:
                    target["masks"] = masks[keep]
                else:
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target["masks"] = target["masks"].bool()

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def crop(image: torch.Tensor, target: Dict[str, Any], region: Tuple[int, int, int, int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def resize(image: torch.Tensor, target: Optional[Dict[str, Any]], size: Union[int, Tuple[int, int], List[int]], max_size: Optional[int] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size: Tuple[int, int], size: int, max_size: Optional[int] = None) -> Tuple[int, int]:
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size: Tuple[int, int], size: Union[int, Tuple[int, int], List[int]], max_size: Optional[int] = None) -> Tuple[int, int]:
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size()[-2:], size, max_size)
    rescaled_image = F.resize(image, size)
    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size()[-2:], image.size()[-2:]))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5


    return rescaled_image, target


def hflip(image: torch.Tensor, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    flipped_image = F.hflip(image)

    w, h = image.size()[-2:][::-1]

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


class Compose(object):
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, image: Any, target: Any) -> Tuple[Any, Any]:
        for t in self.transforms:
            try:
                image, target = t(image, target)
            except Exception as e:
                print(e)
                import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()
        return image, target

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Normalize(object):
    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, target: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class ToTensor(object):
    def __call__(self, img: Union[torch.Tensor, np.ndarray], target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return F.to_tensor(img), target


class RandomHorizontalFlip(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: torch.Tensor, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1: Any, transforms2: Any, p: float = 0.5) -> None:
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img: Any, target: Any) -> Tuple[Any, Any]:
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class SquareResize(object):
    def __init__(self, sizes: List[int]) -> None:
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes

    def __call__(self, img: torch.Tensor, target: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        size = random.choice(self.sizes)
        rescaled_img=F.resize(img, (size, size))
        w, h = rescaled_img.shape[-2:][::-1]
        if target is None:
            return rescaled_img, None
        ratios = tuple(
            float(s) / float(s_orig) for s, s_orig in zip(rescaled_img.shape[-2:][::-1], img.shape[-2:][::-1]))
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        target["size"] = torch.tensor([h, w])

        if "masks" in target:
            target['masks'] = interpolate(
                target['masks'][:, None].float(), (h, w), mode="nearest")[:, 0] > 0.5

        return rescaled_img, target


class RandomResize(object):
    def __init__(self, sizes: List[int], max_size: Optional[int] = None) -> None:
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img: torch.Tensor, target: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int) -> None:
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: torch.Tensor, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        w = random.randint(self.min_size, min(img.shape[-1], self.max_size))
        h = random.randint(self.min_size, min(img.shape[-2], self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


def compute_multi_scale_scales(resolution: int, expanded_scales: bool = False, patch_size: int = 16, num_windows: int = 4) -> List[int]:
    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]  # ensure minimum image size
    return proposed_scales


def make_mujinextended_transforms(image_set: str, resolution: int, multi_scale: bool = False, expanded_scales: bool = False, skip_random_resize: bool = False, patch_size: int = 16, num_windows: int = 4) -> Compose:

    normalize = Compose([
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return Compose([
            ToTensor(),
            RandomHorizontalFlip(),
            SquareResize(scales),
            # RandomSelect(
                # SquareResize(scales),
                # Compose([
                #     RandomResize([400, 500, 600]),
                #     RandomSizeCrop(384, 600),
                #     SquareResize(scales),
                # ])
            # ),
            normalize,
        ])

    if image_set == 'val' or image_set == 'valid' or image_set == 'test':
        return Compose([
            ToTensor(),
            SquareResize([resolution]),
            normalize,
        ])
    if image_set == 'val_speed':
        return Compose([
            ToTensor(),
            SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
