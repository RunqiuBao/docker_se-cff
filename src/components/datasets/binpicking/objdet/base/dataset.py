import numpy
import torch.utils.data
import os
from glob import glob
import json
import cv2
from torch import Tensor
from torchvision.datasets import coco
from pycocotools.mask import frPyObjects, decode


class CocoSegmentation(coco.CocoDetection):
    def __init__(self, root: str, annFile: str, imageHeight: int, imageWidth: int):
        super().__init__(root=root, annFile=annFile, skipImg=True)
        self._imageHeight = imageHeight
        self._imageWidth = imageWidth

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        for category in target:
            seg_rle = category['segmentation']
            category['segmentation'] = decode(frPyObjects(seg_rle, self._imageWidth, self._imageHeight))
        return img, target


class StereoObjDetDataset(torch.utils.data.Dataset):
    path_to_labels = None  # path to the labels folder
    _cocoDataset = None
    NO_VALUE = None
    _isLoadCOCOFormat = None
    _imageSize = None

    def __init__(self, root: str, imageHeight: int, imageWidth: int, isLoadCOCOFormat: bool=False):
        self.NO_VALUE = 0
        self.path_to_labels = os.path.join(root, "annotations.json")
        # Note: need customized torchvision.coco support.
        self._cocoDataset = CocoSegmentation(root=root, annFile=self.path_to_labels, imageHeight=imageHeight, imageWidth=imageWidth)
        self._isLoadCOCOFormat = isLoadCOCOFormat
        self._imageSize = numpy.array([imageWidth, imageHeight])

    def __len__(self):
        return len(self._cocoDataset)

    def __getitem__(self, indexFrame):
        labels_data = self._cocoDataset[indexFrame][1]
        labels_data = self.FormatLabels(labels_data, indexFrame)

        return labels_data

    def FormatLabels(self, labels_data, indexFrame):
        """
        Format the labels:
            - left:  probability map.
            - right:  probability map.
        """
        if self._isLoadCOCOFormat:
            boxes, labels, image_ids, areas, iscrowd, segMaps = {"left": [], "right": []}, {"left": [], "right": []}, {"left": [], "right": []}, {"left": [], "right": []}, {"left": [], "right": []}, {"left": [], "right": []}
            try:
                for target in labels_data:
                    if target['bbox'][0] > self._imageSize[0]:
                        side = "right"
                        oneTargetBox = numpy.array(target['bbox'])
                        oneTargetBox[0] -= self._imageSize[0]
                        boxes[side].append(oneTargetBox[numpy.newaxis, :])
                    else:
                        side = "left"
                        boxes[side].append(numpy.array(target['bbox'])[numpy.newaxis, :])
                    labels[side].append(target['category_id'])
                    # print("category_id: {}".format(target['category_id']))
                    image_ids[side].append(target['image_id'])
                    areas[side].append(target['area'])
                    iscrowd[side].append(target['iscrowd'])
                    segMaps[side].append(target['segmentation'][numpy.newaxis, ...])
                labels_data = {
                    side: {
                        "boxes": numpy.concatenate(boxes[side], axis=0),
                        "labels": numpy.array(labels[side]),
                        "image_id": numpy.array(image_ids[side]),
                        "area": numpy.array(areas[side]),
                        "iscrowd": numpy.array(iscrowd[side]),
                        # "segMaps": numpy.concatenate(segMaps[side], axis=0),
                        "orig_size": self._imageSize,
                        "idx": numpy.array([indexFrame])
                    } for side in ["left", "right"]
                }
            except:
                print("boxes left len: {}".format(len(boxes["left"])))
                print("boxes right len: {}".format(len(boxes["right"])))
                print("baodebug: \n{}".format(boxes["left"]))
                print("baodebug2: \n{}".format(boxes["right"]))
                raise
            return labels_data
        else:
            leftMap = numpy.zeros((self._cocoDataset._imageHeight, self._cocoDataset._imageWidth), dtype='uint8')
            rightMap = numpy.zeros((self._cocoDataset._imageHeight, self._cocoDataset._imageWidth), dtype='uint8')
            for target in labels_data:
                segMask = target['segmentation']
                if numpy.where(segMask > 0)[1].mean() > self._cocoDataset._imageHeight:
                    rightMap = cv2.bitwise_or(rightMap, segMask[:, (self._cocoDataset._imageWidth):])
                else:
                    leftMap = cv2.bitwise_or(leftMap, segMask[:, :(self._cocoDataset._imageWidth)])
            return {
                "pmap": {
                    "left": leftMap,
                    "right": rightMap
                }
            }

    def collate_fn(self, batch):
        """
        batch is a list of dict.
        """
        return batch
