import numpy
import torch.utils.data
import os
import json

from datumaro.components.dataset import Dataset
from datumaro.components.annotation import Bbox, Polygon


cvat_label_to_coco_plus = {
    0: 81,  # container,
    1: 81,  # container,
}


class ObjDetDataset(torch.utils.data.Dataset):
    _path_to_labels = None  # path to the labels folder
    _num_label_files = None

    def __init__(self, root: str, num_repeat: int):
        self._num_repeat = num_repeat
        self._root_path = root

        self._path_to_labels = os.path.join(root, "annotations.xml")
        try:
            self._cvatDataset = Dataset.import_from(self._path_to_labels, format='cvat')
        except:
            import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()
        self._cvatDataset = sorted(self._cvatDataset, key=lambda item: item.id)
        print("====> lenth cvatDataset: ", len(self._cvatDataset))

    def __len__(self):
        return len(self._cvatDataset) * self._num_repeat

    def __getitem__(self, indexFrame: int):
        indexFrame = indexFrame % len(self._cvatDataset)
        try:
            labels_data = self._cvatDataset[indexFrame].annotations
            labels_data = self.FormatLabels(labels_data, indexFrame, self._cvatDataset[indexFrame].media.data, self._cvatDataset[indexFrame].id)
        except Exception as e:
            print("Error in loading labels({}) for frame {}: {}".format(self._path_to_labels, indexFrame, e))
            raise
        return labels_data

    def FormatLabels2(self, labels_data, indexFrame, imageData, imageId):
        """
        format the labels into a dict with 2 keys:
            - 'bbox'
            - 'segmentation'
        """
        outputs = {}
        polygons = []
        for annotation in labels_data:
            if isinstance(annotation, Bbox):
                bbox_xywh = numpy.array(annotation.get_bbox())
                bbox_xyxy = bbox_xywh.copy()
                bbox_xyxy[2] = bbox_xywh[0] + bbox_xywh[2]
                bbox_xyxy[3] = bbox_xywh[1] + bbox_xywh[3]
                area = annotation.get_area()
                outputs[annotation.group] = {
                    'bbox': bbox_xyxy,
                    'category_id': cvat_label_to_coco_plus[annotation.label],
                    'image_id': int(imageId.split("_")[1]),  # file name like 1769495555626_1769495564545_2026-01-27T15_32_44_545, the second number is sensor timestamp.
                    'image_path': os.path.join(self._root_path, "../imagedata/" + imageId + ".pkl"),
                    'area': area,
                }
            elif isinstance(annotation, Polygon):
                if annotation.group in outputs:
                    outputs[annotation.group]['segmentation'] = [annotation.as_polygon()]
                else:
                    polygons.append(annotation)
        for polygon in polygons:
            assert polygon.group in outputs, "bbox for polygon (group {}) not found".format(polygon.group)
            outputs[polygon.group]['segmentation'] = [polygon.as_polygon()]
        return list(outputs.values())
    
    def FormatLabels(self, labels_data, indexFrame, imageData, imageId):
        """
        format the labels into a dict with 2 keys:
            - 'bbox'
        """
        outputs = []
        for annotation in labels_data:
            if isinstance(annotation, Bbox):
                bbox_xywh = numpy.array(annotation.get_bbox())
                bbox_xyxy = bbox_xywh.copy()
                bbox_xyxy[2] = bbox_xywh[0] + bbox_xywh[2]
                bbox_xyxy[3] = bbox_xywh[1] + bbox_xywh[3]
                area = annotation.get_area()
                outputs.append({
                    'bbox': bbox_xyxy,
                    'category_id': cvat_label_to_coco_plus[annotation.label],
                    'image_id': int(imageId.split("_")[1]),  # file name like 1769495555626_1769495564545_2026-01-27T15_32_44_545, the second number is sensor timestamp.
                    'image_path': os.path.join(self._root_path, "../imagedata/" + imageId + ".pkl"),
                    'area': area,
                })
        return outputs

    def collate_fn(self, batch: list):
        """
        batch is a list of dict.
        """
        return batch
