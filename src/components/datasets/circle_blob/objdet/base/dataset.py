import numpy
import torch.utils.data
import os
import json
import cv2

from torch import Tensor


class ObjDetDataset(torch.utils.data.Dataset):
    _path_to_labels = None  # path to the labels folder
    _num_label_files = None

    def __init__(self, root: str, num_repeat: int):
        self._num_repeat = num_repeat
        self._path_to_labels = os.path.join(root, "labels")
        path_to_datalist = os.path.join(root, "../", "datalist.txt")
        self._root_path = root
        with open(path_to_datalist, "r") as datalist_file:
            self._label_files_list = datalist_file.readlines()
            self._label_files_list = [data_file.strip() for data_file in self._label_files_list]
            self._label_files_list.sort()
        assert len(self._label_files_list) > 0
        self._num_label_files = len(self._label_files_list)

    def __len__(self):
        return self._num_label_files * self._num_repeat

    def __getitem__(self, indexData: int):
        indexData = indexData % self._num_label_files
        file_path = os.path.join(self._path_to_labels, self._label_files_list[indexData] + ".json")
        try:
            with open(file_path, "r") as file:
                labels_data = json.load(file)
        except:
            raise FileNotFoundError(f"The label file {file_path} does not exist.")

        try:
            labels_data = self.FormatLabels(labels_data)
        except:
            print("file_path: ", file_path)
            raise

        return labels_data

    def FormatLabels(self, labels_data):
        """
        format the labels into a dict with 2 keys:
            - 'bboxes': Nx5 tensor. 5 including: [
                    X_tl,  # top left corner X in the image
                    Y_tl,  # top left corner Y in the image
                    X_br,  # bottom right corner X in the image
                    Y_br,  # bottom right corner Y in the image
                    index_box  # index in N bboxes.
                ]
            - 'labels': (N,) tensor. classes of the bboxes
        """
        bboxes = []
        labels = []
        for indexInstance, oneInstance in enumerate(labels_data["shapes"]):
            labels.append(numpy.array([int(oneInstance["label"])]))
            bboxes.append(
                numpy.array(
                    [
                        oneInstance["points"][0][0],
                        oneInstance["points"][0][1],
                        oneInstance["points"][1][0],
                        oneInstance["points"][1][1],
                        indexInstance
                    ]
                )[numpy.newaxis, :]
            )
        bboxes = numpy.concatenate(bboxes, axis=0)
        labels = numpy.concatenate(labels)
        return {
            "bboxes": bboxes,
            "labels": labels
        }

    def collate_fn(self, batch: list):
        """
        batch is a list of dict.
        """
        return batch
