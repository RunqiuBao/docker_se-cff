from typing import List, Dict, Tuple

import cv2
import torch
from torch import Tensor
import numpy


def DrawResultBboxesAndKeyptsOnStereoEventFrame(left_event_sharp, right_event_sharp, sbboxes, classes, confidences, keypts1, keypts2, stereo_confidences=None):
    if keypts1.ndim == 1:
        keypts1 = keypts1.unsqueeze(0)
        keypts2 = keypts2.unsqueeze(0)
    
    left_event_sharp = left_event_sharp - left_event_sharp.min()
    left_event_sharp = (left_event_sharp * 255 / left_event_sharp.max()).astype('uint8')
    left_event_sharp = cv2.cvtColor(left_event_sharp, cv2.COLOR_GRAY2BGR)
    right_event_sharp = right_event_sharp - right_event_sharp.min()
    right_event_sharp = (right_event_sharp * 255 / right_event_sharp.max()).astype('uint8')
    right_event_sharp = cv2.cvtColor(right_event_sharp, cv2.COLOR_GRAY2BGR)
    for ii, (bbox, classindex, confidence, keypt1, keypt2) in enumerate(zip(sbboxes, classes, confidences, keypts1, keypts2)):
        top_left = (int(bbox[0]), int(bbox[1]))
        top_right = (int(bbox[2]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(left_event_sharp, top_left, bottom_right, (255, 0, 0), thickness=3)
        text = 'confi: {:.4f}'.format(confidence.item())
        textposition = (int(top_right[0] + bottom_right[0]) // 2, int(top_right[1] + bottom_right[1]) // 2)
        cv2.putText(left_event_sharp, text, textposition, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
        w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_right[1]
        keypt1_int = (int(keypt1[0] * w + top_left[0]), int(keypt1[1] * h + top_left[1]))
        keypt2_int = (int(keypt2[0] * w + top_left[0]), int(keypt2[1] * h + top_left[1]))        
        cv2.circle(left_event_sharp, keypt1_int, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(left_event_sharp, keypt2_int, radius=5, color=(0, 0, 255), thickness=-1)

        top_left = (int(bbox[4]), int(bbox[1]))
        top_right = (int(bbox[5]), int(bbox[1]))
        bottom_right = (int(bbox[5]), int(bbox[3]))
        cv2.rectangle(right_event_sharp, top_left, bottom_right, (255, 0, 0), thickness=3)
        if stereo_confidences is not None:
            text = 'confi: {:.4f}'.format(stereo_confidences[ii].item())
            textposition = (int(top_right[0] + bottom_right[0]) // 2, int(top_right[1] + bottom_right[1]) // 2)
            cv2.putText(right_event_sharp, text, textposition, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
        w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_right[1]
        keypt1_int_r = (int(keypt1[0] * w + top_left[0]), int(keypt1[1] * h + top_left[1]))
        keypt2_int_r = (int(keypt2[0] * w + top_left[0]), int(keypt2[1] * h + top_left[1]))        
        cv2.circle(right_event_sharp, keypt1_int_r, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(right_event_sharp, keypt2_int_r, radius=5, color=(0, 0, 255), thickness=-1)
    return left_event_sharp, right_event_sharp


def RenderImageWithBboxesAndKeypts(
    left_event_sharp: numpy.ndarray,
    right_event_sharp: numpy.ndarray,
    obj_preds: Dict
) -> Tuple[Tensor]:
    """
    Args:
        left_event_sharp: ...
    """
    sbboxes, classes, confidences, keypts1, keypts2 = obj_preds['sbboxes'], obj_preds['classes'], obj_preds['confidences'], obj_preds['keypt1s'], obj_preds['keypt2s']
    visz_left, visz_right = DrawResultBboxesAndKeyptsOnStereoEventFrame(
        left_event_sharp,
        right_event_sharp,
        sbboxes,
        classes,
        confidences,
        keypts1,
        keypts2
    )
    return torch.from_numpy(visz_left), torch.from_numpy(visz_right)


def RenderImageWithBboxes(
    left_event_sharp: numpy.ndarray,
    obj_preds: Dict,
    is_output_torch: bool=True
):
    """
    Args:
        left_event_sharp: ...
    """
    bboxes, classes = obj_preds['bboxes'], obj_preds['classes']    
    confidences = obj_preds.get('confidences', torch.ones_like(classes) * -1)
    
    left_event_sharp = left_event_sharp - left_event_sharp.min()
    left_event_sharp = (left_event_sharp * 255 / left_event_sharp.max()).astype('uint8')
    left_event_sharp = cv2.cvtColor(left_event_sharp, cv2.COLOR_GRAY2RGB)    
    for bbox, classindex, confidence in zip(bboxes, classes, confidences):
        top_left = (int(bbox[0]), int(bbox[1]))
        top_right = (int(bbox[2]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(left_event_sharp, top_left, bottom_right, (255, 0, 0), thickness=1)
        text = 'cls: {}\nconfi: {}'.format(format(classindex.item(), '.2f'), format(confidence.item(), '.2f'))
        textposition = (int(top_right[0] + bottom_right[1]) // 2, int(top_right[1] + bottom_right[1]) // 2)
        cv2.putText(left_event_sharp, text, textposition, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
        w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_right[1]
    if is_output_torch:
        return [torch.from_numpy(left_event_sharp),]
    else:
        return [left_event_sharp]
