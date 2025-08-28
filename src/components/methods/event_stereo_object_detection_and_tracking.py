import os.path
import torch
import cv2
import torchvision
from torchvision.ops import nms
import time
from tqdm import tqdm

from ..models.utils.misc import DetachCopyNested
from .base import batch_to_cuda
from ..models.yolo_pose_utils import non_max_suppression
from .event_stereo_object_detection_with_yolo_pose import FilterBadDetections, FilterTemporal, SaveTestResultsAndVisualize

import logging
logger = logging.getLogger(__name__)


@torch.no_grad()
def test(
    models,
    data_loader,
    sequence_name,
    save_root,
    is_save_onnx = False
):
    for model in models.values():
        model.module.eval()

    if is_save_onnx:
        logger.info(
            '''
            # how to do inference with the exported onnx model:
            # providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            import onnxruntime
            providers = ["CPUExecutionProvider"]
            ort_session = onnxruntime.InferenceSession(os.path.join(save_root, "concentration_net.onnx"), providers=providers)
            ort_inputs = {"left_event": oneInputs["event"]["left"].cpu().numpy(), "right_event": oneInputs["event"]["right"].cpu().numpy()}
            ort_outs = ort_session.run(["left_event_sharp", "right_event_sharp"], ort_inputs)
            '''
        )
    
    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    previous_preds = None
    prediction_dict = None
    for indexBatch in range(len(data_loader.dataset)):
        batch_data = batch_to_cuda(next(data_iter))
        starttime = time.time()
        # ---------- concentration net ----------
        left_event_sharp = models["concentration_net"].module.predict(batch_data["event"]["left"])
        right_event_sharp = models["concentration_net"].module.predict(batch_data["event"]["right"])
        if is_save_onnx:
            torch.onnx.export(
                models['concentration_net'].module,
                (
                    batch_data["event"]["left"],
                    batch_data["event"]["right"]
                ),
                os.path.join(save_root, "concentration_net.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["left_img", "right_img"],
                output_names=["left_preds", "right_preds"]
            )

        imageHeight, imageWidth = batch_data["event"]["left"].shape[-2:]
        batch_img_metas = {"h": imageHeight, "w": imageWidth}
        num_classes = models["objdet_head"].module.config["num_classes"]

        # ---------- disp pred net ----------
        pred_disparity_pyramid = models["disp_head"].module.predict(left_event_sharp, right_event_sharp)
        if is_save_onnx:
            torch.onnx.export(
                models['disp_head'].module,
                (
                    left_event_sharp,
                    right_event_sharp
                ),
                os.path.join(save_root, "disp_head.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["left_img", "right_img"],
                output_names=["preds"]
            )

        # ---------- objdet net ----------
        left_detections = models["objdet_head"].module.predict(batch_data["event"]["left"])
        if is_save_onnx:
            torch.onnx.export(
                models['objdet_head'].module,
                (
                    batch_data["event"]["left"],
                ),
                os.path.join(save_root, "objdet_head.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["left_event_voxel", "right_event_voxel"],
                output_names=["preds0", "preds100", "preds101", "preds102", "preds11", "artifacts00", "artifacts01", "artifacts02"]
            )

        left_detections_multilevels_detachcopy = DetachCopyNested(left_detections)
        left_bboxesClsKeypts_nmsed_topked, nms_topk_mask = non_max_suppression(
            left_detections_multilevels_detachcopy,
            conf_thres=models["objdet_head"].module.config["confidence_threshold_inference"],
            iou_thres=models["objdet_head"].module.config["nms_iou_threshold_inference"],
            labels=[],
            nc=models["objdet_head"].module.config["num_classes"],
            multi_label=False,
            agnostic=False,
            max_det=models["objdet_head"].module.config["num_topk_candidates"],
            end2end=False,
        )

        mask_track_nonbackground = None
        if prediction_dict:
            # ---------- local tracking head ----------
            previous_detections = prediction_dict['objdet'][0]
            device, dtype = previous_detections.device, previous_detections.dtype
            (
                batch_track_bboxes_priors,
                batch_corresponding_previousdet_ids,
                batch_track_refined_bboxes,
                batch_track_refined_scores,
                batch_track_predicted_keypts,
                rpn_cls_scores,
                rpn_bbox_preds,
            ) = models["local_tracking_head"].module.predict(
                batch_data["event"]["left"],
                [previous_detections[..., :4]],
                torch.zeros((1, imageHeight, imageWidth), device=device, dtype=dtype),
                batch_img_metas
            )
            
            if batch_track_bboxes_priors[0] is not None:
                (
                    mask_track_nonbackground,
                    tracked_bboxes_nobkg,
                    tracked_scores,
                    tracked_keypts_nobkg
                ) = models["local_tracking_head"].module.extract_inference_results(
                    batch_track_bboxes_priors[0].squeeze(0),
                    batch_track_refined_bboxes[0].view(-1, num_classes, 4),
                    batch_track_refined_scores[0],
                    batch_track_predicted_keypts[0].view(-1, num_classes, models["local_tracking_head"].module.config["max_num_keypoints"] * 3)
                )
                corresponding_previousdets_indices = batch_corresponding_previousdet_ids[0][mask_track_nonbackground]
                corresponding_previousdets = previous_detections[corresponding_previousdets_indices]
                tracked_class_labels = corresponding_previousdets[:, 8].unsqueeze(-1)
                tracked_confidences = corresponding_previousdets[:, 9].unsqueeze(-1)

        refined_sbboxes_nobkg = None
        if left_bboxesClsKeypts_nmsed_topked[0].shape[0] > 0:
            # ---------- stereo detection head ----------
            left_bboxes_nmsed_topked = [one_batch[..., :4] for one_batch in left_bboxesClsKeypts_nmsed_topked]
            left_confidences_nmsed_topked = torch.max(left_bboxesClsKeypts_nmsed_topked[0][:, 4:(4 + num_classes)], dim=-1)[0].unsqueeze(-1)
            if mask_track_nonbackground is not None and mask_track_nonbackground.sum() > 0:
                left_bboxes_nmsed_topked[0] = torch.concat([left_bboxes_nmsed_topked[0], tracked_bboxes_nobkg[:, :4]], dim=0)
                left_confidences = left_confidences_nmsed_topked + 1.0  # prioritize current detections.
                keep_indices = nms(
                    left_bboxes_nmsed_topked[0],
                    torch.cat([left_confidences, tracked_confidences], dim=0).squeeze(1),
                    iou_threshold=models["objdet_head"].module.config["nms_iou_threshold_inference"],
                )
                keep_mask = torch.zeros(left_bboxes_nmsed_topked[0].size(0), dtype=torch.bool, device=left_bboxes_nmsed_topked[0].device)
                keep_mask[keep_indices] = True
                keep_mask_fromtracked = keep_mask[left_confidences.shape[0]:]
                tracked_class_labels = tracked_class_labels[keep_mask_fromtracked]
                tracked_confidences = tracked_confidences[keep_mask_fromtracked]
                left_bboxes_nmsed_topked[0] = left_bboxes_nmsed_topked[0][keep_mask]

            (
                batch_sbboxes_priors,
                batch_corresponding_leftdet_ids,
                batch_refined_right_bboxes,
                batch_refined_right_scores,
                batch_predicted_right_keypts,
                rpn_cls_scores,
                rpn_bbox_preds,
            ) = models["stereo_detection_head"].module.predict(
                batch_data["event"]["right"],
                left_bboxes_nmsed_topked,
                pred_disparity_pyramid[-1],
                batch_img_metas,
            )
            if is_save_onnx:
                torch.onnx.export(
                    models['stereo_detection_head'].module,
                    (
                        batch_data["event"]["right"],
                        left_bboxes_nmsed_topked,
                        pred_disparity_pyramid[-1],
                        batch_img_metas,
                    ),
                    os.path.join(save_root, "stereo_detection_head.onnx"),
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=["right_feat", "left_bboxes", "disp_prior", "batch_img_metas"],
                    output_names=["batch_sbboxes_priors", "batch_refined_right_bboxes", "batch_refined_right_scores", "batch_predicted_right_keypts", "rpn_cls_scores", "rpn_bbox_preds"],
                )

            assert left_event_sharp.shape[0] == 1  # batch size should be 1
            # select best right bboxes and keypts for visualization.
            (
                mask_nonbackground,
                refined_sbboxes_nobkg,
                refined_right_scored_pred,
                right_keypts_pred_nobkg
            ) = models["stereo_detection_head"].module.extract_inference_results(
                batch_sbboxes_priors[0].squeeze(0),
                batch_refined_right_bboxes[0].view(-1, num_classes, 4),
                batch_refined_right_scores[0],
                batch_predicted_right_keypts[0].view(-1, num_classes, models["stereo_detection_head"].module.config["max_num_keypoints"] * 3)
            )

        logger.info("one infer time: {} sec.".format(time.time() - starttime))
        if is_save_onnx and (left_bboxesClsKeypts_nmsed_topked[0].shape[0] > 0):
            print("==================================== finished onnx model (event_stereo_object_detection_with_yolo_pose) export! ====================================")
            break

        if refined_sbboxes_nobkg is not None:
            # (l_tl_x, l_tl_y, l_br_x, l_br_y,
            #                                  r_tl_x, r_tl_y, r_br_x, r_br_y,
            #                                                                 class_label, confidence, confidence_right,
            #                                                                                                           l_kpt0_x, l_kpt0_y, visibility_l0, l_kpt1_x, l_kpt1_y, visibility_l1, ..., r_kpt0_x, r_kpt0_y, visibility_r0, r_kpt1_x, r_kpt1_y, visibility_r1, ...)
            # TODO: mark right confidences on the result visz image.
            # TODO: do not use seeds. use left and detected, tracked.
            num_left_detected = left_confidences_nmsed_topked.shape[0]
            mask_left_detected = batch_corresponding_leftdet_ids[0] < num_left_detected
            corresponding_left_detected_indices = batch_corresponding_leftdet_ids[0][mask_left_detected][mask_nonbackground[mask_left_detected]]
            corresponding_leftdets = left_bboxesClsKeypts_nmsed_topked[0][corresponding_left_detected_indices]
            left_bboxes_final = corresponding_leftdets[:, 0:4]
            left_classlabels_final = torch.argmax(corresponding_leftdets[:, 4:(4 + num_classes)], dim=-1).unsqueeze(-1)
            left_confidences_final = torch.max(corresponding_leftdets[:, 4:(4 + num_classes)], dim=-1)[0].unsqueeze(-1)
            left_keypts_final = corresponding_leftdets[:, (4 + models["objdet_head"].module.config["num_classes"]):]
            if mask_track_nonbackground is not None and mask_track_nonbackground.sum() > 0 and keep_mask_fromtracked.sum() > 0:
                mask_left_tracked = ~mask_left_detected
                corresponding_left_tracked_indices = batch_corresponding_leftdet_ids[0][mask_left_tracked][mask_nonbackground[mask_left_tracked]]
                corresponding_left_tracked_indices -= num_left_detected  # rebase to tracked indices
                leftdets_tracked = corresponding_previousdets[keep_mask_fromtracked][corresponding_left_tracked_indices]

                left_bboxes_final = torch.concat([left_bboxes_final, leftdets_tracked[:, :4]], dim=0)
                left_classlabels_final = torch.concat([
                    left_classlabels_final,
                    leftdets_tracked[:, 8].unsqueeze(-1),
                ], dim=0)
                left_confidences_final = torch.concat([
                    left_confidences_final,
                    leftdets_tracked[:, 9].unsqueeze(-1),
                ], dim=0)
                left_keypts_final = torch.concat([
                    left_keypts_final,
                    leftdets_tracked[:, 11:17].view(-1, models["stereo_detection_head"].module.config["max_num_keypoints"] * 3)
                ])

            # # -------------- debug code --------------
            # dummy_results = batch_sbboxes_priors[0][0, :, 16, :]
            # dummy_results = torch.concat([
            #     dummy_results[:, :4],
            #     dummy_results[:, 4].unsqueeze(-1),
            #     dummy_results[:, 1].unsqueeze(-1),
            #     dummy_results[:, 5].unsqueeze(-1),
            #     dummy_results[:, 3].unsqueeze(-1)
            # ], dim=-1)
            # dummy_results = torch.concat([
            #     dummy_results,
            #     torch.ones((dummy_results.shape[0], 3), device=dummy_results.device, dtype=dummy_results.dtype),
            #     torch.zeros((dummy_results.shape[0], 12), device=dummy_results.device, dtype=dummy_results.dtype),
            # ], dim=-1)
            # preds = dummy_results
            # # -------------- debug code --------------

            raw_preds = torch.concat([
                left_bboxes_final,
                torch.concat([
                    refined_sbboxes_nobkg[:, 4].view(-1, 1),
                    left_bboxes_final[:, 1].view(-1, 1),
                    refined_sbboxes_nobkg[:, 6].view(-1, 1),
                    left_bboxes_final[:, 3].view(-1, 1)
                ], dim=-1),
                left_classlabels_final,
                left_confidences_final,
                refined_right_scored_pred.view(-1, 1),
                left_keypts_final,
                right_keypts_pred_nobkg.view(-1, models["stereo_detection_head"].module.config["max_num_keypoints"] * 3),
            ], dim=1)
            # align the stereo keypts in y
            raw_preds[:, (12 + models["stereo_detection_head"].module.config["max_num_keypoints"] * 3):(12 + models["stereo_detection_head"].module.config["max_num_keypoints"] * 3 * 2):3] = raw_preds[:, 12:(12 + models["stereo_detection_head"].module.config["max_num_keypoints"] * 3):3]
            right_keep_indices = nms(
                raw_preds[:, 4:8],
                raw_preds[:, 10],
                iou_threshold=models["stereo_detection_head"].module.config["right_nms_iou_threshold_inference"]
            )
            raw_preds = raw_preds[right_keep_indices]
            left_keep_indices = nms(
                raw_preds[:, 0:4],
                raw_preds[:, 9],
                iou_threshold=models["objdet_head"].module.config["confidence_threshold_inference"]
            )
            raw_preds = raw_preds[left_keep_indices]

            preds = FilterBadDetections(
                raw_preds,
                imageHeight=batch_data["image_metadata"]["h_recti"],
                imageWidth=batch_data["image_metadata"]["w_recti"],
                margin=0,
                right_confidence_threshold=models["stereo_detection_head"].module.config["right_confidence_threshold_inference"],
                left_right_confidence_diff=models["stereo_detection_head"].module.config["left_right_confidence_diff_inference"],
                left_right_width_diff_threshold=models["stereo_detection_head"].module.config["left_right_width_diff_threshold"],
            )
            # preds = FilterIrregularBboxes(preds, hw_ratiorange_class0=[1.9, 3.15])
            # preds = FilterTemporal(
            #     preds,
            #     previous_preds,
            #     iou_threshold_for_matching=0.4,
            #     iou_leftright_for_filtering=0.7,
            #     area_change_threshold=0.7
            # )

            if preds is not None:
                # final nms
                # keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
                preds_track = preds[preds[:, 9] > 1.0]
                keep_indices = torchvision.ops.nms(preds_track[:, :4], preds_track[:, 9], models["objdet_head"].module.config["nms_iou_threshold_inference"])
                preds = torch.concat([
                    preds[preds[::, 9] <= 1.0],
                    preds_track[keep_indices]
                ])

                prediction_dict = {
                    "objdet": [preds],
                    "concentrate": {
                        "left": left_event_sharp,
                        "right": right_event_sharp
                    },
                    "ts": batch_data["end_timestamp"].item(),
                    "disp": cv2.cvtColor(pred_disparity_pyramid[-1].detach().cpu().numpy().astype('uint8')[0], cv2.COLOR_GRAY2BGR)
                }
                stereo_visz = SaveTestResultsAndVisualize(
                    prediction_dict,
                    indexBatch,
                    batch_data["end_timestamp"].item(),
                    sequence_name,
                    save_root,
                    batch_data["image_metadata"]
                )
                # # -------------- debug code --------------
                # os.makedirs("/root/data/debug_test/", exist_ok=True)
                # h, w = stereo_visz[0].shape[:2]
                # h = h // 2
                # cv2.imwrite("/root/data/debug_test/" + str(previous_prediction_dict['ts']) + ".png", numpy.vstack([previous_prediction_dict['disp'][:h, :w], stereo_visz[0]]))
                # # -------------- debug code --------------

                previous_preds = preds
            else:
                prediction_dict = None
                logger.error("batch {} has no valid detections.".format(indexBatch))
        else:
            prediction_dict = None
            logger.error("batch {} has no valid detections.".format(indexBatch))

        pbar.update(1)
    pbar.close()
    return
