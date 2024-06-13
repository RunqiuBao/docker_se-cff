import numpy
import torch.utils.data
import os
from glob import glob
import json


class StereoObjDetDataset(torch.utils.data.Dataset):
    path_to_labels = None  # path to the labels folder
    num_label_files = None

    def __init__(self, root):
        self.path_to_labels = os.path.join(root, "labels")
        label_files_list = glob(os.path.join(self.path_to_labels, "*.json"))
        self.num_label_files = len(label_files_list)
        pass

    def __len__(self):
        return self.num_label_files

    def __getitem__(self, timestamp):
        file_path = os.path.join(self.path_to_labels, str(timestamp).zfill(8) + ".json")
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
                    delta_y_keypt2  # normalized Y distance of keypt2 from top left corner at left image
                ]
            - 'labels': (N,) tensor. classes of the bboxes
        """
        bboxes = []
        labels = []
        for oneInstance in labels_data["shapes"]:
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
                        (oneInstance["keypt1"][0] - oneInstance["leftPoints"][0][0])
                        / (
                            oneInstance["leftPoints"][1][0]
                            - oneInstance["leftPoints"][0][0]
                        ),
                        (oneInstance["keypt1"][1] - oneInstance["leftPoints"][0][1])
                        / (
                            oneInstance["leftPoints"][1][1]
                            - oneInstance["leftPoints"][0][1]
                        ),
                        (x_keypt2 - oneInstance["leftPoints"][0][0])
                        / (
                            oneInstance["leftPoints"][1][0]
                            - oneInstance["leftPoints"][0][0]
                        ),
                        (y_keypt2 - oneInstance["leftPoints"][0][1])
                        / (
                            oneInstance["leftPoints"][1][1]
                            - oneInstance["leftPoints"][0][1]
                        ),
                    ]
                )[numpy.newaxis, :]
            )
        return {
            "bboxes": numpy.concatenate(bboxes, axis=0),
            "labels": numpy.concatenate(labels),
        }

    def collate_fn(self, batch):
        """
        batch is a list of dict.
        """
        return batch
