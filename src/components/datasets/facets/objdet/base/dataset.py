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
            if len(seg_rle) > 0:
                category['segmentation'] = decode(frPyObjects(seg_rle, self._imageHeight, self._imageWidth * 2))
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
        self._cocoDataset = CocoSegmentation(root=root, annFile=self.path_to_labels, imageHeight=imageHeight, imageWidth=imageWidth * 2)
        self._isLoadCOCOFormat = isLoadCOCOFormat
        self._imageSize = numpy.array([imageWidth, imageHeight])
        print("baodebug: event frame size: {}".format(self._imageSize))

    def __len__(self):
        return len(self._cocoDataset)

    def __getitem__(self, indexFrame):
        labels_data = self._cocoDataset[indexFrame][1]
        # print("frame ({}), labels_data_0: {}".format(indexFrame, labels_data[0]["bbox"]))
        labels_data = self.FormatLabels(labels_data, indexFrame)

        return labels_data

    def FormatLabels(self, labels_data, indexFrame):
        """
        format the labels into a dict with 2 keys:
            - 'bboxes': Nx7 tensor. 7 including: [
                    X_tl,  # top left corner X at left image
                    Y_tl,  # top left corner Y at left image
                    X_br,  # bottom right corner X at left image
                    Y_br,  # bottom right corner Y at left image
                    X_tl_r,  # top left corner X at right image
                    X_br_r,  # bottom right corner X at right image
                    index_box  # index in N bboxes.
                ]
            - 'labels': (N,) tensor. classes of the bboxes
            - '*masks': N elements list. segmentation masks.
        """
        if self._isLoadCOCOFormat:
            bboxes = []
            labels = []
            leftmasks = []
            labels_formatted = {}
            # rightmasks = []
            left_targets = [target for target in labels_data if (target["bbox"][0] + target["bbox"][2] / 2) < self._imageSize[0]]
            left_targets = sorted(left_targets, key=lambda x: x["bbox"][0] + x["bbox"][2] / 2)
            right_targets = [target for target in labels_data if (target["bbox"][0] + target["bbox"][2] / 2) >= self._imageSize[0]]
            right_targets = sorted(right_targets, key=lambda x: x["bbox"][0] + x["bbox"][2] / 2)
            try:
                assert len(left_targets) == len(right_targets)
            except:
                print("baodebug: frame (" + str(indexFrame) + ") left targets " + str(len(left_targets)) + ", right targets " + str(len(right_targets)))
                raise
            try:
                for indexTarget in range(len(left_targets)):
                    if left_targets[indexTarget]["category_id"] != right_targets[indexTarget]["category_id"]:
                        print("!!Error: frame ({}) stereo targets class indexNotMatch: left {}, right {}".format(indexFrame, left_targets[indexTarget]["category_id"], right_targets[indexTarget]["category_id"]))
                        continue
                    labels.append(numpy.array([int(left_targets[indexTarget]["category_id"])]))
                    x_center = left_targets[indexTarget]["bbox"][0] + left_targets[indexTarget]["bbox"][2] / 2
                    y_center = left_targets[indexTarget]["bbox"][1] + left_targets[indexTarget]["bbox"][3] / 2
                    x_center_r = right_targets[indexTarget]["bbox"][0] + right_targets[indexTarget]["bbox"][2] / 2
                    enlarge_factor = 1.0
                    w_l = left_targets[indexTarget]["bbox"][2] * enlarge_factor
                    h_l = left_targets[indexTarget]["bbox"][3] * enlarge_factor
                    w_r = right_targets[indexTarget]["bbox"][2] * enlarge_factor
                    X_tl = numpy.clip(x_center - w_l / 2, 0, self._imageSize[0])
                    Y_tl = numpy.clip(y_center - h_l / 2, 0, self._imageSize[1])
                    X_br = numpy.clip(x_center + w_l / 2, 0, self._imageSize[0])
                    Y_br = numpy.clip(y_center + h_l / 2, 0, self._imageSize[1])
                    X_tl_r = numpy.clip(x_center_r - w_r / 2 - self._imageSize[0], 0, self._imageSize[0])
                    X_br_r = numpy.clip(x_center_r + w_r / 2 - self._imageSize[0], 0, self._imageSize[0])
                    bboxes.append(
                        numpy.array(
                            [
                                X_tl,
                                Y_tl,
                                X_br,
                                Y_br,
                                X_tl_r,
                                X_br_r,
                                indexTarget
                            ]
                        )[numpy.newaxis, :]
                    )
                    cocoSegMap = left_targets[indexTarget]["segmentation"]
                    leftmask = cocoSegMap[:, :self._imageSize[0]].squeeze() if isinstance(cocoSegMap, numpy.ndarray) else numpy.zeros((self._imageSize[1], self._imageSize[0]))
                    leftmasks.append(leftmask)
                    # rightmask = right_targets[indexTarget]["segmentation"]
                    # rightmasks.append(None if isinstance(rightmask, list) else rightmask.squeeze())

                    # # test finding quad corners
                    # contours = cv2.findContours((leftmasks[0] * 255).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                    # print("arcLength: {}".format(cv2.arcLength(contours, True)))
                    # epsilon = 0.02 * cv2.arcLength(contours, True)
                    # approx = cv2.approxPolyDP(contours, epsilon, True)
                    # print("approx: {}".format(approx))
                    # debugImg = cv2.cvtColor((leftmasks[0] * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
                    # for point in approx:
                    #     cv2.circle(debugImg, (point[0][0], point[0][1]), radius=3, color=(0, 255, 0), thickness=-1)
                    # cv2.imwrite("/root/data/debug.png", debugImg)
                if bboxes:
                    bboxes = numpy.concatenate(bboxes, axis=0)
                    labels_formatted["bboxes"] = bboxes
                if labels:
                    labels = numpy.concatenate(labels) if labels else labels
                    labels_formatted["labels"] = labels
                if leftmasks:
                    labels_formatted["leftmasks"] = leftmasks
                    # "rightmasks": rightmasks
            except:
                raise
            return labels_formatted
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
