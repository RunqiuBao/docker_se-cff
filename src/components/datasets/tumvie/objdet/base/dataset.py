import numpy
import torch.utils.data
import os
from glob import glob
import json
import cv2
from torch import Tensor
from datumaro.components.dataset import Dataset
from datumaro.components.annotation import Polygon, Bbox, Points
from collections import defaultdict


def FormatKeypoints(keyPoints: numpy.ndarray, max_num_keypoints: int) -> numpy.ndarray:
    """
    If max_num_keypoints is 1, average all the key points and return as (1, 3); Else, return the key points as (N, 3) shape.
    If keyPoints number is less than max_num_keypoints, pad with zeros.
    Args:
        keyPoints: (N, 2) shape

    Returns:
        keyPoints: (N, 3) shape.
    """
    keyPoints = [[corner[0], corner[1], 2] for corner in keyPoints]
    if max_num_keypoints == 1:
        keyPoints = numpy.mean(keyPoints, axis=0)[None, :]
    else:
        keyPoints = numpy.array(keyPoints)
        if keyPoints.shape[0] < max_num_keypoints:
            keyPoints = numpy.pad(
                keyPoints,
                ((0, max_num_keypoints - keyPoints.shape[0]), (0, 0)),
                mode='constant',
                constant_values=0
            )
    return keyPoints


class StereoObjDetDataset(torch.utils.data.Dataset):
    path_to_labels = None  # path to the labels folder
    _cvatDataset = None
    NO_VALUE = None
    _isLoadCOCOFormat = None
    _imageSize = None
    is_initilized = False
    _num_label_files = None
    _timestamps = None

    def __init__(self, root: str, imageHeight: int, imageWidth: int, num_repeat: int=1, timestamps=None, max_num_keypoints=None, **kwargs):
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
            self._max_num_keypoints = max_num_keypoints if max_num_keypoints is not None else 1
        except Exception as e:
            print("objdet annotations loading failed: {}".format(e))
            try:
                self.path_to_labels = os.path.join(root, "labels")
                label_files_list = glob(os.path.join(self.path_to_labels, "*.json"))
                assert len(label_files_list) > 0
                self._num_data_files = len(label_files_list)
                self._num_label_files = self._num_data_files * self._num_repeat
                self._imageSize = numpy.array([imageWidth, imageHeight])
                self.is_initilized = True
                self._timestamps = timestamps
            except Exception as e:
                if kwargs["dataset_type"] != "test":
                    print("!!!!!Error!!, objdet annotations loading failed again: {}".format(e))

    def __len__(self):
        return self._num_label_files

    def __getitem__(self, indexFrame):
        if not self.is_initilized:
            return

        indexFrame = indexFrame % self._num_data_files
        if self._cvatDataset is not None:
            try:
                labels_data = self._cvatDataset[indexFrame].annotations
                # print("frame ({}), labels_data_0: {}".format(indexFrame, labels_data[0]["bbox"]))
                labels_data = self.FormatLabels(labels_data, indexFrame, self._cvatDataset[indexFrame].media.data, self._cvatDataset[indexFrame].id)
                labels_data["timestamp"] = self._cvatDataset[indexFrame].id
            except Exception as e:
                print("Error in loading labels({}) for frame {}: {}".format(self.path_to_labels, indexFrame, e))
                raise
        else:
            timestamp = self._timestamps[indexFrame]
            file_path = os.path.join(self.path_to_labels, str(timestamp).zfill(12) + ".json")
            try:
                with open(file_path, "r") as file:
                    labels_data = json.load(file)
            except:
                raise FileNotFoundError(f"The label file {file_path} does not exist.")

            labels_data = self.FormatLabels2(labels_data)
            labels_data["timestamp"] = str(timestamp)

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
            if len(oneGroup) > 2: # containing separate points.
                left_target_group, right_target_group = {}, {}
                for oneTargetInOneGroup in oneGroup:
                    if isinstance(oneTargetInOneGroup, Points):
                        if (oneTargetInOneGroup.points[0] + oneTargetInOneGroup.points[2]) / 2 < self._imageSize[0]:
                            left_target_group["points"] = oneTargetInOneGroup
                        else:
                            right_target_group["points"] = oneTargetInOneGroup
                    if isinstance(oneTargetInOneGroup, Bbox):
                        if (oneTargetInOneGroup.get_bbox()[0] + oneTargetInOneGroup.get_bbox()[2] / 2) < self._imageSize[0]:
                            left_target_group["bbox"] = oneTargetInOneGroup
                        else:
                            right_target_group["bbox"] = oneTargetInOneGroup
                left_targets.append(left_target_group)
                right_targets.append(right_target_group)
            else:
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
                left_target = left_targets[indexTarget]
                right_target = right_targets[indexTarget]
                if isinstance(left_target, dict):
                    left_target_points = left_target["points"]
                    left_target = left_target["bbox"]
                    right_target_points = right_target["points"]
                    right_target = right_target["bbox"]
                else:
                    left_target_points, right_target_points = None, None

                if left_target.label != right_target.label:
                    print("!!Error: frame ({}) stereo targets class indexNotMatch: left {}, right {}".format(indexFrame, left_target.label, right_target.label))
                    continue

                labels.append(numpy.array([int(left_target.label)]))
                left_bbox = left_target.get_bbox()  # Note: format [x_min, y_min, w, h]
                right_bbox = right_target.get_bbox()
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

                if isinstance(left_target, Polygon):
                    left_corners = numpy.array(left_target.points).reshape(-1, 2)
                    leftcorners.append(
                        FormatKeypoints(left_corners, self._max_num_keypoints)[None, :]
                    )

                    right_corners = numpy.array(right_target.points).reshape(-1, 2)
                    for ii in range(right_corners.shape[0]):
                        right_corners[ii][0] -= self._imageSize[0]
                    rightcorners.append(
                        FormatKeypoints(right_corners, self._max_num_keypoints)[None, :]
                    )
                elif left_target_points is not None:
                    # Bbox with separate key points
                    left_corners = numpy.array(left_target_points.points).reshape(-1, 2)
                    leftcorners.append(
                        FormatKeypoints(left_corners, self._max_num_keypoints)[None, :]
                    )
                    right_corners = numpy.array(right_target_points.points).reshape(-1, 2)
                    rightcorners.append(
                        FormatKeypoints(right_corners, self._max_num_keypoints)[None, :]
                    )
                else:
                    # Bbox without separate key points
                    left_bbox = left_target.get_bbox()  # Note: format [x_min, y_min, w, h]
                    leftcorners.append(
                        FormatKeypoints(
                            numpy.array([left_bbox[0] + left_bbox[2] / 2, left_bbox[1] + left_bbox[3] / 2, 2])[None, :],
                            self._max_num_keypoints
                        )[None, :]
                    )
                    right_bbox = right_target.get_bbox()
                    rightcorners.append(
                        FormatKeypoints(
                            numpy.array([right_bbox[0] + right_bbox[2] / 2 - self._imageSize[0], right_bbox[1] + right_bbox[3] / 2, 2])[None, :],
                            self._max_num_keypoints
                        )[None, :]
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
    
    def FormatLabels2(self, labels_data):
        """
        For blender-vibration dataset.
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
        bboxes, keypts_left, keypts_right = [], [], []
        labels = []
        for indexInstance, oneInstance in enumerate(labels_data["shapes"]):
            labels.append(numpy.array([int(oneInstance["label"])]))
            x_keypt2 = (oneInstance["keypt2"][0][0] + oneInstance["keypt2"][1][0]) / 2
            y_keypt2 = (oneInstance["keypt2"][0][1] + oneInstance["keypt2"][1][1]) / 2
            bboxes.append(
                numpy.array(
                    [
                        oneInstance["leftPoints"][0][0],
                        oneInstance["leftPoints"][0][1],
                        oneInstance["leftPoints"][1][0],
                        oneInstance["leftPoints"][1][1],
                        oneInstance["rightPoints"][0][0],
                        oneInstance["rightPoints"][1][0],
                        indexInstance
                    ]
                )[numpy.newaxis, :]
            )
            disparity = (oneInstance["leftPoints"][0][0] + oneInstance["leftPoints"][1][0]) / 2 - (oneInstance["rightPoints"][0][0] + oneInstance["rightPoints"][1][0]) / 2
            leftcorner_oneinstance = numpy.array([
                [oneInstance["keypt1"][0], oneInstance["keypt1"][1], 2],
                [x_keypt2, y_keypt2, 2]
            ])
            rightcorner_oneinstance = leftcorner_oneinstance.copy()
            keypts_left.append(
                leftcorner_oneinstance[None, :]
            )
            rightcorner_oneinstance[:, 0] = rightcorner_oneinstance[:, 0] - disparity
            keypts_right.append(
                rightcorner_oneinstance[None, :]
            )
        if bboxes:
            bboxes = numpy.concatenate(bboxes, axis=0)
            labels = numpy.concatenate(labels)
            keypts_left = numpy.concatenate(keypts_left, axis=0)
            keypts_right = numpy.concatenate(keypts_right, axis=0)
        return {
            "bboxes": bboxes,
            "labels": labels,
            "keypts": keypts_left,
            "keypts_right": keypts_right
        }

    def collate_fn(self, batch):
        """
        batch is a list of dict.
        """
        return batch
