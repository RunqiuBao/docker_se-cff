import os.path
import onnxruntime
import numpy
import torch
import torch.nn.functional as F
import pytest
import time
import cv2
import pickle

from components.models.yolo_pose_utils import non_max_suppression
from components.models.utils.objdet_utils import WarpBboxes
from components.models.utils.objdet_utils import AllocateHypothesesToTargets
from components.models.utils.objdet_utils import ExtractStereoInferenceResults


@pytest.mark.parametrize("onnx_path", ["/root/code/docker_pytorch_trainnn/experiments/binpicking"])
def test_executeseod(onnx_path: str):
    providers = ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'CPUExecutionProvider']
    networkNames = [
       "concentration_net",
       "disp_head",
       "objdet_head",
       "stereo_detection_rpn_head",
       "stereo_detection_head",
    ]
    onnxSessions = {}
    for networkName in networkNames:
        onnxSessions[networkName] = onnxruntime.InferenceSession(
            os.path.join(onnx_path, networkName + ".onnx"), providers=providers
        )
    
    # load sample of input data
    with open(os.path.join(onnx_path, "leftRightEvents.pkl"), 'rb') as f:
        data = pickle.load(f)
    left_right_events = data["left_right_events"]
    imageHeight, imageWidth = left_right_events.shape[-2:]
    num_classes = 1
    max_num_keypoints = 2

    starttime = time.time()
    (left_right_sharps,) = onnxSessions['concentration_net'].run(None, {'left_right_inputs': left_right_events})
    print(f"concentration net time cost: {time.time() - starttime}")

    starttime = time.time()
    inputsDispHead = {
        'left_img': left_right_sharps[0][None, :],
        'right_img': left_right_sharps[1][None, :],
    }
    pred_disparity_pyramid = onnxSessions['disp_head'].run(None, inputsDispHead)
    print(f"disphead net time cost: {time.time() - starttime}")

    starttime = time.time()
    left_detections = onnxSessions['objdet_head'].run(None, {'left_event_voxel': left_right_events[0][None, :]})
    print(f"objdet net time cost: {time.time() - starttime}")

    starttime = time.time()
    left_bboxesClsKeypts_nmsed_topked, nms_topk_mask = non_max_suppression(
        [torch.from_numpy(oneElement) for oneElement in left_detections],
        conf_thres=0.5,
        iou_thres=0.2,
        labels=[],
        nc=1,
        multi_label=False,
        agnostic=False,
        max_det=8,
        end2end=False,
    )
    print(f"nms time cost: {time.time() - starttime}")

    if left_bboxesClsKeypts_nmsed_topked and left_bboxesClsKeypts_nmsed_topked[0].shape[0] > 0:
        starttime = time.time()
        left_bboxes_nmsed_topked = [one_batch[..., :4] for one_batch in left_bboxesClsKeypts_nmsed_topked]
        inputsStereoRPNHead = {
            'right_event_voxel': left_right_events[1][None, :],
        }
        stereo_rpn_outputs = onnxSessions['stereo_detection_rpn_head'].run(None, inputsStereoRPNHead)
        print(f"stereo rpn time cost: {time.time() - starttime}")

        starttime = time.time()
        rpn_bbox_preds = [stereo_rpn_outputs[0], stereo_rpn_outputs[1], stereo_rpn_outputs[2], stereo_rpn_outputs[3]]
        rpn_cls_scores = [stereo_rpn_outputs[4], stereo_rpn_outputs[5], stereo_rpn_outputs[6], stereo_rpn_outputs[7]]
        right_feats = [stereo_rpn_outputs[8], stereo_rpn_outputs[9], stereo_rpn_outputs[10], stereo_rpn_outputs[11]]
        warped_left_bboxes = WarpBboxes(
            left_bboxes_nmsed_topked,
            torch.from_numpy(pred_disparity_pyramid[-1]),
            imageHeight,
            imageWidth,
        )
        batch_rpn_hypotheses = AllocateHypothesesToTargets(
            [torch.from_numpy(oneElement) for oneElement in rpn_cls_scores],
            [torch.from_numpy(oneElement) for oneElement in rpn_bbox_preds],
            warped_left_bboxes,
            imageHeight,
            imageWidth,
            num_classes=num_classes,
            nms_pred=200,
            max_hypotheses_per_img=100,
            min_iou_with_target=0.5,
        )
        num_rpn_hypotheses = batch_rpn_hypotheses[0].get_dict()["bboxes"].shape[0]
        import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()
        if num_rpn_hypotheses < 20:
            hypotheses_bboxes = F.pad(batch_rpn_hypotheses[0].get_dict()["bboxes"], (0, 0, 0, 20 - num_rpn_hypotheses), "constant", 0)
            hypotheses_target_ids = F.pad(batch_rpn_hypotheses[0].get_dict()["target_ids"], (0, 20 - num_rpn_hypotheses), "constant", 0)
        else:
            hypotheses_bboxes = batch_rpn_hypotheses[0].get_dict()["bboxes"][:20]
            hypotheses_target_ids = batch_rpn_hypotheses[0].get_dict()["target_ids"][:20]
        inputsStereoHead = {
            'right_feats0': right_feats[0],
            'right_feats1': right_feats[1],
            'right_feats2': right_feats[2],
            'right_feats3': right_feats[3],
            'left_bboxes': left_bboxes_nmsed_topked[0].numpy(),
            'hypotheses_bboxes': hypotheses_bboxes.numpy(),
            'hypotheses_target_ids': hypotheses_target_ids.numpy(),
        }
        print(f"stereo preparation time cost: {time.time() - starttime}")
        starttime = time.time()
        (
            sbboxes_priors,
            target_ids,
            right_bboxes_refine,
            cls_score,
            right_keypts_pred,
        ) = onnxSessions['stereo_detection_head'].run(None, inputsStereoHead)
        print(f"stereo net time cost: {time.time() - starttime}")
        starttime = time.time()
        (
            mask_nonbackground,
            refined_sbboxes_nobkg,
            refined_right_scored_pred,
            right_keypts_pred_nobkg,
        ) = ExtractStereoInferenceResults(
            torch.from_numpy(sbboxes_priors).squeeze(0),
            torch.from_numpy(right_bboxes_refine).view(-1, num_classes, 4),
            torch.from_numpy(cls_score),
            torch.from_numpy(right_keypts_pred).view(-1, num_classes, max_num_keypoints * 3),
            2,
        )
        print(f"extract time cost: {time.time() - starttime}")

        img_left = left_right_sharps[0, 0]
        img_left -= img_left.min()
        img_left = img_left * 255 / img_left.max()
        img_right = left_right_sharps[1, 0]
        img_right -= img_right.min()
        img_right = img_right * 255 / img_right.max()
        for i in range(refined_sbboxes_nobkg.shape[0]):
            # Extract Left Box
            l_box = refined_sbboxes_nobkg[i, :4].numpy().astype("int")
            cv2.rectangle(img_left, (l_box[0], l_box[1]), (l_box[2], l_box[3]), (0, 255, 0), 2)
            
            # Extract Right Box
            r_box = refined_sbboxes_nobkg[i, 4:].numpy().astype("int")
            cv2.rectangle(img_right, (r_box[0], r_box[1]), (r_box[2], r_box[3]), (255, 0, 0), 2)

        # Optional: Concatenate images to see them side-by-side
        combined = numpy.hstack((img_left, img_right))
        cv2.imwrite(os.path.join(onnx_path, "setere_detection_onnx.png"), combined)
