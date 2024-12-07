from typing import List, Dict, Tuple

import os
import copy
import cv2
import torch
from torch import Tensor
import numpy


def LimitBboxWithInImage(bbox: numpy.ndarray, imageHeight: int, imageWidth: int):
    newBbox = numpy.ones_like(bbox)
    newBbox[0::2] = numpy.clip(bbox[0::2], 0, imageWidth)
    newBbox[5] = numpy.clip(bbox[5], 0, imageWidth)
    newBbox[[1, 3]] = numpy.clip(bbox[[1, 3]], 0, imageHeight)
    return newBbox


def draw_featmap_on_view(top_left, bottom_right, featmap, viewImg, enlarge_featmap_factor):
    """
    Args:
        top_left: top left corner of the bbox,
        bottom_right: bottom right corner of the bbox.
        featmap: feature map (the shape is normalized. need to resize to bbox shape).
        viewImg: the viewImg that has same size as camera image.
        enlarge_featmap_factor: factor to enlarge the featmap wrt. bbox. As the featmap may be predicted with an enlarged bbox.
    """
    w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
    if enlarge_featmap_factor is not None:
        w *= enlarge_featmap_factor
        h *= enlarge_featmap_factor
    center_x = (bottom_right[0] + top_left[0]) / 2
    center_y = (bottom_right[1] + top_left[1]) / 2
    segMap = numpy.clip(featmap * 255, 0, 255).astype('uint8')
    imageHeight, imageWidth = viewImg.shape[:2]
    try:
        top_y = max(int(center_y - h / 2), 0)
        bottom_y = min(int(center_y + h / 2), imageHeight)
        h = max(bottom_y - top_y, h)
        top_x = max(int(center_x - w / 2), 0)
        bottom_x = min(int(center_x + w / 2), imageWidth)
        w = max(bottom_x - top_x, w)
        mask_patch = cv2.resize(segMap, (int(w), int(h)))
        if (bottom_y - top_y) < int(h):
            if int(center_y - h / 2) <= 0:
                mask_patch_bottom_y = int(h)
                mask_patch_top_y = mask_patch_bottom_y - (bottom_y - top_y)
            elif int(center_y + h / 2) >= imageHeight:
                mask_patch_top_y = 0
                mask_patch_bottom_y = (bottom_y - top_y)
        else:
            mask_patch_top_y = 0
            mask_patch_bottom_y = int(h)
        if (bottom_x - top_x) < int(w):
            if int(center_x - w / 2) <= 0:
                mask_patch_bottom_x = int(w)
                mask_patch_top_x = mask_patch_bottom_x - (bottom_x - top_x)
            elif int(center_x + w / 2) >= imageWidth:
                mask_patch_top_x = 0
                mask_patch_bottom_x = (bottom_x - top_x)
        else:
            mask_patch_top_x = 0
            mask_patch_bottom_x = int(w)
        viewImg[top_y:bottom_y, top_x:bottom_x] = mask_patch[mask_patch_top_y:mask_patch_bottom_y, mask_patch_top_x:mask_patch_bottom_x]
    except:
        from IPython import embed; print('here!'); embed()
    return viewImg


def draw_corners_on_view(top_left, bottom_right, featmap, viewImg, enlarge_featmap_factor, indexDet, indexImg, side):
    """
    Find the four corners in the contour. And draw on the viewImg.
    """
    segMap = numpy.zeros((viewImg.shape[0], viewImg.shape[1]), dtype="uint8")
    # featmap = 1 - numpy.clip(featmap, 0, 1)
    featmap[numpy.where(featmap < 0.5)] *= 0  # filter on noise.
    segMap = draw_featmap_on_view(top_left, bottom_right, featmap, segMap, enlarge_featmap_factor)
    # smoothing
    kernel = numpy.ones((5, 5), dtype="uint8")
    segMap = cv2.erode(segMap, kernel, iterations = 1)
    cv2.imwrite("/root/code/docker_pytorch_trainnn/experiments/facets/inference/det_visz/" + str(indexImg).zfill(6) + "_{}_{}.png".format(indexDet, side), segMap)
    featContours = cv2.findContours(segMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    largest_area = 0
    for contour in featContours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            featContour = contour
            largest_area = area
    if largest_area == 0:
        print("Error finding contours. featContours: {}".format(featContours))
        return viewImg
    print("arcLength: {}".format(cv2.arcLength(featContour, True)))
    epsilon = 0.01 * cv2.arcLength(featContour, True)
    approx = cv2.approxPolyDP(featContour, epsilon, True)
    print("approx: {}".format(approx))
    # for point in approx:
    #     cv2.circle(viewImg, (point[0][0], point[0][1]), radius=4, color=(0, 255, 0), thickness=2)
    # if len(approx) == 4:
    cv2.polylines(viewImg, [featContour], isClosed=True, color=(0, 255, 0), thickness=2)
    return viewImg


def DrawResultBboxesAndKeyptsOnStereoEventFrame(
        left_event_sharp,
        right_event_sharp,
        sbboxes,
        classes,
        confidences,
        keypts1=None,
        keypts2=None,
        facets=None,
        facets_right=None,
        enlarge_facet_factor=None,
        stereo_confidences=None,
        indexBatch=None
):
    if keypts1 is not None and keypts1.ndim == 1:
        keypts1 = keypts1.unsqueeze(0)
        keypts2 = keypts2.unsqueeze(0)
    if isinstance(facets, torch.Tensor):
        facets = facets.numpy()
    if isinstance(facets_right, torch.Tensor):
        facets_right = facets_right.numpy()
    
    left_event_sharp = left_event_sharp - left_event_sharp.min()
    left_event_sharp = (left_event_sharp * 255 / left_event_sharp.max()).astype('uint8')
    left_event_sharp = cv2.cvtColor(left_event_sharp, cv2.COLOR_GRAY2BGR)
    right_event_sharp = right_event_sharp - right_event_sharp.min()
    right_event_sharp = (right_event_sharp * 255 / right_event_sharp.max()).astype('uint8')
    right_event_sharp = cv2.cvtColor(right_event_sharp, cv2.COLOR_GRAY2BGR)
    imageHeight, imageWidth = left_event_sharp.shape[:2]
    if facets is not None:
        instances_facets = numpy.zeros_like(left_event_sharp, dtype='uint8')
    if facets_right is not None:
        instances_facets_right = numpy.zeros_like(right_event_sharp, dtype='uint8')
    for ii, (bbox, classindex, confidence) in enumerate(zip(sbboxes, classes, confidences)):
        bbox = LimitBboxWithInImage(bbox, *left_event_sharp.shape[:2])
        top_left = (int(bbox[0]), int(bbox[1]))
        top_right = (int(bbox[2]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        if True:#facets is None and keypts1 is None:
            cv2.rectangle(left_event_sharp, top_left, bottom_right, (255, 0, 0), thickness=3)
        text = 'cf:{:.4f},cl:{}'.format(confidence.item(), classindex)
        textposition = (int(top_right[0] + bottom_right[0]) // 2, int(top_right[1] + bottom_right[1]) // 2)
        cv2.putText(left_event_sharp, text, textposition, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
        w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_right[1]
        if keypts1 is not None:
            keypt1, keypt2 = keypts1[ii], keypts2[ii]
            keypt1_int = (int(keypt1[0] * w + top_left[0]), int(keypt1[1] * h + top_left[1]))
            keypt2_int = (int(keypt2[0] * w + top_left[0]), int(keypt2[1] * h + top_left[1]))        
            cv2.circle(left_event_sharp, keypt1_int, radius=5, color=(0, 255, 0), thickness=-1)
            cv2.circle(left_event_sharp, keypt2_int, radius=5, color=(0, 0, 255), thickness=-1)
        if facets is not None:
            # instances_facets = cv2.cvtColor(instances_facets, cv2.COLOR_BGR2GRAY)
            # instances_facets = draw_featmap_on_view(top_left, bottom_right, facets[ii], instances_facets, enlarge_facet_factor)
            # instances_facets = cv2.cvtColor(instances_facets, cv2.COLOR_GRAY2BGR)
            instances_facets = draw_corners_on_view(top_left, bottom_right, facets[ii], instances_facets, enlarge_facet_factor, ii, indexBatch, "left")

        top_left = (int(bbox[4]), int(bbox[1]))
        top_right = (int(bbox[5]), int(bbox[1]))
        bottom_right = (int(bbox[5]), int(bbox[3]))
        if True:#facets is None and keypts1 is None:
            cv2.rectangle(right_event_sharp, top_left, bottom_right, (255, 0, 0), thickness=3)
        if stereo_confidences is not None:
            text = 'cf:{:.4f},cl:{}'.format(stereo_confidences[ii].item(), classindex)
            textposition = (int(top_right[0] + bottom_right[0]) // 2, int(top_right[1] + bottom_right[1]) // 2)
            cv2.putText(right_event_sharp, text, textposition, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
        w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_right[1]
        if keypts1 is not None:
            keypt1_int_r = (int(keypt1[0] * w + top_left[0]), int(keypt1[1] * h + top_left[1]))
            keypt2_int_r = (int(keypt2[0] * w + top_left[0]), int(keypt2[1] * h + top_left[1]))        
            cv2.circle(right_event_sharp, keypt1_int_r, radius=5, color=(0, 255, 0), thickness=-1)
            cv2.circle(right_event_sharp, keypt2_int_r, radius=5, color=(0, 0, 255), thickness=-1)
        if facets_right is not None:
            # instances_facets_right = cv2.cvtColor(instances_facets_right, cv2.COLOR_BGR2GRAY)
            # instances_facets_right = draw_featmap_on_view(top_left, bottom_right, facets_right[ii], instances_facets_right, enlarge_facet_factor)
            # instances_facets_right = cv2.cvtColor(instances_facets_right, cv2.COLOR_GRAY2BGR)
            instances_facets_right = draw_corners_on_view(top_left, bottom_right, facets[ii], instances_facets_right, enlarge_facet_factor, ii, indexBatch, "right")
    if facets is not None:
        left_event_sharp = cv2.addWeighted(left_event_sharp, 0.5, (255 - instances_facets), 0.5, 0)
    if facets_right is not None:
        right_event_sharp = cv2.addWeighted(right_event_sharp, 0.5, (255 - instances_facets_right), 0.5, 0)
    return left_event_sharp, right_event_sharp


def RenderImageWithBboxesAndKeypts(
    left_event_sharp: numpy.ndarray,
    right_event_sharp: numpy.ndarray,
    obj_preds: Dict,
    is_enable_keypt: bool = False,
    is_enable_facet: bool = True
) -> Tuple[Tensor]:
    """
    Args:
        left_event_sharp: ...
    """
    sbboxes, classes, confidences = obj_preds['sbboxes'], obj_preds['classes'], obj_preds['confidences']
    keyptsAndFeats = {}
    if 'keypt1s' in obj_preds:
        keyptsAndFeats['keypts1'] = obj_preds['keypt1s']
        keyptsAndFeats['keypts2'] = obj_preds['keypt2s']
    if 'facets' in obj_preds:
        keyptsAndFeats["facets"] = obj_preds["facets"]
    if 'facets_right' in obj_preds:
        keyptsAndFeats["facets_right"] = obj_preds["facets_right"]
    visz_left, visz_right = DrawResultBboxesAndKeyptsOnStereoEventFrame(
        left_event_sharp,
        right_event_sharp,
        sbboxes,
        classes,
        confidences,
        **keyptsAndFeats
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
    bboxes, classes, segMaps = obj_preds['bboxes'], obj_preds['classes'], obj_preds.get('segMaps', None)
    confidences = obj_preds.get('confidences', torch.ones_like(classes) * -1)

    left_event_sharp = left_event_sharp - left_event_sharp.min()
    left_event_sharp = (left_event_sharp * 255 / left_event_sharp.max()).astype('uint8')
    left_event_sharp = cv2.cvtColor(left_event_sharp, cv2.COLOR_GRAY2RGB)
    imageHeight, imageWidth = left_event_sharp.shape[:2]
    if segMaps is not None:  
        instances_segmap = numpy.zeros_like(left_event_sharp, dtype='uint8')
    for indexInstance, (bbox, classindex, confidence) in enumerate(zip(bboxes, classes, confidences)):
        top_left = (max(min(int(bbox[0]), imageWidth), 0), max(min(int(bbox[1]), imageHeight), 0))
        top_right = (max(min(int(bbox[2]), imageWidth), 0), max(min(int(bbox[1]), imageHeight), 0))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(left_event_sharp, top_left, bottom_right, (255, 0, 0), thickness=1)
        text = 'cls: {}\nconfi: {}'.format(format(classindex.item(), '.2f'), format(confidence.item(), '.2f'))
        textposition = (int(top_right[0] + bottom_right[1]) // 2, int(top_right[1] + bottom_right[1]) // 2)
        cv2.putText(left_event_sharp, text, textposition, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
        w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_right[1]
        if segMaps is not None and (w > 0) and (h > 0):
            segMap = numpy.clip(segMaps[indexInstance] * 255, 0, 255).astype('uint8')
            # # hack: fix when black white inverted
            # # put instance masks in each bounding boxes
            # segMap = cv2.threshold(segMap, 200, 255, cv2.THRESH_BINARY)[-1]
            # imageSize = segMap.shape[0]
            # if numpy.mean(segMap[imageSize // 3:2 * imageSize // 3, imageSize // 3:2 * imageSize // 3]) < 127:
            #     segMap = 255 - segMap
            try:
                instances_segmap[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cv2.cvtColor(cv2.resize(segMap, (w, h), cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
            except:
                from IPython import embed; embed()
    if segMaps is not None:
        left_event_sharp = cv2.addWeighted(left_event_sharp, 0.5, (255 - instances_segmap), 0.5, 0)

    if is_output_torch:
        return [torch.from_numpy(left_event_sharp),]
    else:        
        return [left_event_sharp]
