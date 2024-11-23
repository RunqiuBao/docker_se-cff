import numpy
import torch.utils.data
import os
from glob import glob
import json
import cv2

from torch import Tensor


class StereoObjDetDataset(torch.utils.data.Dataset):
    path_to_labels = None  # path to the labels folder
    num_label_files = None
    is_initilized = False

    def __init__(self, root, img_height, img_width):
        try:
            self.path_to_labels = os.path.join(root, "labels")
            label_files_list = glob(os.path.join(self.path_to_labels, "*.json"))
            assert len(label_files_list) > 0
            self.num_label_files = len(label_files_list)
            self.img_height = img_height
            self.img_width = img_width
            self.is_initilized = True
        except:
            pass

    def __len__(self):
        return self.num_label_files

    def __getitem__(self, timestamp):
        if not self.is_initilized:
            return

        file_path = os.path.join(self.path_to_labels, str(timestamp).zfill(12) + ".json")
        try:
            with open(file_path, "r") as file:
                labels_data = json.load(file)
        except:
            raise FileNotFoundError(f"The label file {file_path} does not exist.")

        labels_data = self.FormatLabels(labels_data)
        return labels_data

    def FormatLabels(self, labels_data):
        """
        format the labels into a dict with 2 keys:
            - 'bboxes': Nx10 tensor. 10 including: [
                    X_tl,  # top left corner X at left image
                    Y_tl,  # top left corner Y at left image
                    X_br,  # bottom right corner X at left image
                    Y_br,  # bottom right corner Y at left image
                    X_tl_r,  # top left corner X at right image
                    X_br_r,  # bottom right corner X at right image
                    delta_x_keypt1,  # normalized X distance of keypt1 from top left corner at left image
                    delta_y_keypt1,  # normalized Y distance of keypt1 from top left corner at left image
                    delta_x_keypt2,  # normalized X distance of keypt2 from top left corner at left image
                    delta_y_keypt2,  # normalized Y distance of keypt2 from top left corner at left image
                    index_box  # index in N bboxes.
                ]
            - 'labels': (N,) tensor. classes of the bboxes
        """
        bboxes = []
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
                        (oneInstance["keypt1"][0] - oneInstance["leftPoints"][0][0]) / (oneInstance["leftPoints"][1][0] - oneInstance["leftPoints"][0][0]),
                        (oneInstance["keypt1"][1] - oneInstance["leftPoints"][0][1]) / (oneInstance["leftPoints"][1][1] - oneInstance["leftPoints"][0][1]),
                        (x_keypt2 - oneInstance["leftPoints"][0][0]) / (oneInstance["leftPoints"][1][0] - oneInstance["leftPoints"][0][0]),
                        (y_keypt2 - oneInstance["leftPoints"][0][1]) / (oneInstance["leftPoints"][1][1] - oneInstance["leftPoints"][0][1]),
                        indexInstance
                    ]
                )[numpy.newaxis, :]
            )
        bboxes = numpy.concatenate(bboxes, axis=0)
        labels = numpy.concatenate(labels)
        keypt1_masks = self.GetGtKeyptDistanceMasks(bboxes[:, :4], bboxes[:, 6:10], 1)
        keypt2_masks = self.GetGtKeyptDistanceMasks(bboxes[:, :4], bboxes[:, 6:10], 2)
        return {
            "bboxes": bboxes,
            "labels": labels,
            "keypt1_masks": keypt1_masks,
            "keypt2_masks": keypt2_masks
        }
    
    def GetGtKeyptDistanceMasks(self, bboxes: Tensor, keypts: Tensor, indexKeypt: int) -> Tensor:
        """
        within each bounding box, compute each pixel's distance towards keypoint.

        Args:
            bboxes: shape (num_bboxes, 4). gt bboxes of each detection.
            keypts: shape (num_bboxes, 4). key points in each gt bbox.

        Returns:
            gtMasks: shape (num_bboxes, h, w)
        """
        keypts_in_img = numpy.stack(
            [
                keypts[:, 0] * (bboxes[:, 2] - bboxes[:, 0]) + bboxes[:, 0],
                keypts[:, 1] * (bboxes[:, 3] - bboxes[:, 1]) + bboxes[:, 1],
                keypts[:, 2] * (bboxes[:, 2] - bboxes[:, 0]) + bboxes[:, 0],
                keypts[:, 3] * (bboxes[:, 3] - bboxes[:, 1]) + bboxes[:, 1]
            ], axis=1)
        gtMasks = []
        for indexBbox, keypt_all in enumerate(keypts_in_img):
            if indexKeypt == 1:
                keypt = keypt_all[0:2]
            else:
                keypt = keypt_all[2:4]
            oneMask = numpy.zeros((self.img_height, self.img_width), dtype='uint8')
            top_left = (int(bboxes[indexBbox, 0]), int(bboxes[indexBbox, 1]))
            bottom_right = (int(bboxes[indexBbox, 2]), int(bboxes[indexBbox, 3]))
            oneMask = cv2.rectangle(oneMask, top_left, bottom_right, (1,), -1)
            maskedPoints = numpy.where(oneMask > 0)
            oneMask = oneMask.astype('float')
            oneMask[maskedPoints] *= numpy.sqrt(numpy.power((maskedPoints[1] - keypt[0]), 2) + numpy.power((maskedPoints[0] - keypt[1]), 2))
            oneMask[maskedPoints] = 1 - oneMask[maskedPoints] / oneMask.max()  # Note: do this after roi align.
            # crop by triangle
            triangleMask = numpy.zeros((self.img_height, self.img_width), dtype='uint8')
            bottom_left = (int(bboxes[indexBbox, 0]), int(bboxes[indexBbox, 3]))
            trianglePoints = numpy.array([keypt_all[0:2].astype('int'), bottom_left, bottom_right])
            cv2.fillPoly(triangleMask, [trianglePoints], 255)
            oneMask[~triangleMask.view('bool')] *= 0
            gtMasks.append(oneMask)
        return numpy.stack(gtMasks, axis=0)

    def collate_fn(self, batch):
        """
        batch is a list of dict.
        """
        return batch
