import numpy
import torch.utils.data
import os
from glob import glob
import json
import cv2
from torch import Tensor
from datumaro.components.dataset import Dataset
from datumaro.components.annotation import Polygon, Bbox
from collections import defaultdict


class StereoObjDetDataset(torch.utils.data.Dataset):
    path_to_labels = None  # path to the labels folder
    _cvatDataset = None
    NO_VALUE = None
    _isLoadCOCOFormat = None
    _imageSize = None
    is_initilized = False
    _num_label_files = None

    def __init__(self, root: str, imageHeight: int, imageWidth: int, num_repeat: int=1, **kwargs):
        self._num_repeat = num_repeat  # for data augmentation. Repeating the data sequence and do randomcrop.
        self.NO_VALUE = 0
        try:
            self.path_to_labels = os.path.join(root, "annotations.xml")
            self._cvatDataset = Dataset.import_from(self.path_to_labels, format='cvat')
            print("====> baodebug lenth cvatDataset: ", len(self._cvatDataset))
            self._imageSize = numpy.array([imageWidth, imageHeight])
            self._num_data_files = len(self._cvatDataset)
            self._num_label_files = len(self._cvatDataset) * self._num_repeat  # Note: data augmentation by random crop and repeat
            self.is_initilized = True
        except Exception as e:
            print("objdet annotations loading failed: {}".format(e))
            pass

    def __len__(self):
        return self._num_label_files

    def __getitem__(self, indexFrame):
        if not self.is_initilized:
            return

        indexFrame = indexFrame % self._num_data_files
        try:
            labels_data = self._cvatDataset[indexFrame].annotations
            # print("frame ({}), labels_data_0: {}".format(indexFrame, labels_data[0]["bbox"]))
            labels_data = self.FormatLabels(labels_data, indexFrame, self._cvatDataset[indexFrame].media.data, self._cvatDataset[indexFrame].id)
            labels_data["timestamp"] = self._cvatDataset[indexFrame].id
        except Exception as e:
            print("Error in loading labels({}) for frame {}: {}".format(self.path_to_labels, indexFrame, e))
            raise

        return labels_data

    def FormatLabels(self, labels_data, indexFrame, imageData, imageId):
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
            - 'keypts'
            - 'keypts_right'
        """
        bboxes = []
        labels = []
        leftcorners = []  # N * (4, 2)
        rightcorners = []
        labels_formatted = {}
        # rightmasks = []
        left_targets, right_targets = [], []
        indicesGroup = [annotation.group for annotation in labels_data]
        groupedAnnotations = defaultdict(list)
        for idx, annotation in zip(indicesGroup, labels_data):
            groupedAnnotations[idx].append(annotation)
        for oneGroup in groupedAnnotations.values():
            for oneTargetInOneGroup in oneGroup:
                bboxThisGroup = oneTargetInOneGroup.get_bbox()
                if (bboxThisGroup[0] + bboxThisGroup[2] / 2) < self._imageSize[0]:
                    left_targets.append(oneTargetInOneGroup)
                else:
                    right_targets.append(oneTargetInOneGroup)
        
        # # -------- test code --------
        # debug_path = "/root/data/debug_tumvie_dataload/"
        # os.makedirs(debug_path, exist_ok=True)
        # for indexTarget in range(len(left_targets)):
        #     color = tuple(numpy.random.randint(0, 256, size=3).tolist())
        #     left_bbox = numpy.array(left_targets[indexTarget].get_bbox()).astype('int')
        #     right_bbox = numpy.array(right_targets[indexTarget].get_bbox()).astype('int')
        #     cv2.rectangle(imageData, (left_bbox[0], left_bbox[1]), (left_bbox[0] + left_bbox[2], left_bbox[1] + left_bbox[3]) , color, 2)
        #     cv2.rectangle(imageData, (right_bbox[0], right_bbox[1]), (right_bbox[0] + right_bbox[2], right_bbox[1] + right_bbox[3]) , color, 2)
        # cv2.imwrite(os.path.join(debug_path, "frame_{}_{}_left.png".format(indexFrame, imageId)), imageData)
        # # -------- test code --------

        try:
            assert len(left_targets) == len(right_targets)
        except:
            print("baodebug: frame (" + str(indexFrame) + ") left targets " + str(len(left_targets)) + ", right targets " + str(len(right_targets)))
            print("left_targets:\n")
            for left_target in left_targets:
                print(left_target)
            print("right_targets:\n")
            for right_target in right_targets:
                print(right_target)
            raise
        try:
            for indexTarget in range(len(left_targets)):
                if left_targets[indexTarget].label != right_targets[indexTarget].label:
                    print("!!Error: frame ({}) stereo targets class indexNotMatch: left {}, right {}".format(indexFrame, left_targets[indexTarget].label, right_targets[indexTarget].label))
                    continue
                labels.append(numpy.array([int(left_targets[indexTarget].label)]))
                left_bbox = left_targets[indexTarget].get_bbox()  # Note: format [x_min, y_min, w, h]
                right_bbox = right_targets[indexTarget].get_bbox()
                x_center = left_bbox[0] + left_bbox[2] / 2
                y_center = left_bbox[1] + left_bbox[3] / 2
                x_center_r = right_bbox[0] + right_bbox[2] / 2
                enlarge_factor = 1.0
                w_l = left_bbox[2] * enlarge_factor
                h_l = left_bbox[3] * enlarge_factor
                w_r = right_bbox[2] * enlarge_factor
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

                if isinstance(left_targets[indexTarget], Polygon):
                    left_corners = numpy.array(left_targets[indexTarget].points).reshape(-1, 2)  # Note: format [x_min, y_min, w, h]
                    leftcorners.append(
                        numpy.mean(
                            numpy.array([
                                [left_corners[0][0], left_corners[0][1], 2],
                                [left_corners[1][0], left_corners[1][1], 2],
                                [left_corners[2][0], left_corners[2][1], 2],
                                [left_corners[3][0], left_corners[3][1], 2]
                            ]),
                            axis=0
                        )[None, None, :]
                    )
                    right_corners = numpy.array(right_targets[indexTarget].points).reshape(-1, 2)  # Note: format [x_min, y_min, w, h]
                    rightcorners.append(
                        numpy.mean(
                            numpy.array([
                                [right_corners[0][0] - self._imageSize[0], right_corners[0][1], 2],
                                [right_corners[1][0] - self._imageSize[0], right_corners[1][1], 2],
                                [right_corners[2][0] - self._imageSize[0], right_corners[2][1], 2],
                                [right_corners[3][0] - self._imageSize[0], right_corners[3][1], 2]
                            ]),
                            axis=0
                        )[None, None, :]
                    )
                else:
                    # Bbox
                    left_bbox = left_targets[indexTarget].get_bbox()  # Note: format [x_min, y_min, w, h]
                    leftcorners.append(
                        numpy.array([
                            left_bbox[0] + left_bbox[2] / 2, left_bbox[1] + left_bbox[3] / 2, 2
                        ])[None, None, :]
                    )
                    right_bbox = right_targets[indexTarget].get_bbox()
                    rightcorners.append(
                        numpy.array([
                            right_bbox[0] + right_bbox[2] / 2 - self._imageSize[0], right_bbox[1] + right_bbox[3] / 2, 2
                        ])[None, None, :]
                    )
                    
            if bboxes:
                bboxes = numpy.concatenate(bboxes, axis=0)
                labels_formatted["bboxes"] = bboxes
            if labels:
                labels = numpy.concatenate(labels) if labels else labels
                labels_formatted["labels"] = labels
            if leftcorners:
                labels_formatted["keypts"] = numpy.concatenate(leftcorners, axis=0)
                labels_formatted["keypts_right"] = numpy.concatenate(rightcorners, axis=0)
                # "rightmasks": rightmasks
        except:
            raise
        return labels_formatted

    def collate_fn(self, batch):
        """
        batch is a list of dict.
        """
        return batch
