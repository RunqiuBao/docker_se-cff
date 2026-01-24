import os.path
import torch
import cv2
import copy
from torchmetrics.detection import MeanAveragePrecision
import torchvision
from torchvision.ops import nms
import time
import numpy
import pickle
from tqdm import tqdm

from ..models.utils.misc import DetachCopyNested
from .base import batch_to_cuda
from ..models.yolo_pose_utils import non_max_suppression
from .event_stereo_object_detection_with_yolo_pose import FilterBadDetections, FilterTemporal, SaveTestResultsAndVisualize
from ..models.utils.objdet_utils import evaluate_results_with_gt

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
    prediction_dict = None
    start_collect_onnx = False  # start collecting when enough detections emerged.
    batch_data_for_onnx = None
    infer_time = []
    num_final_detections = {}
    evals_input = {
        "TP": [],
        "FP": [],
        "FN": [],
        "GT": [],
    }
    iou_threshold = 0.75
    for indexBatch in range(len(data_loader.dataset)):
        batch_data = batch_to_cuda(next(data_iter))
        if not batch_data['event'] or batch_data['event'].get('left') is None:
            pbar.update(1)
            logger.warning("batch {} has no event data.".format(indexBatch))
            continue
        
        if is_save_onnx and start_collect_onnx and batch_data_for_onnx is not None:
            batch_data = batch_data_for_onnx
        
        starttime = time.time()
        # ---------- concentration net ----------
        start_subtime = time.time()
        left_event_sharp = models["concentration_net"].module.predict(batch_data["event"]["left"])
        print("concentration_net costs: {} sec.".format(time.time() - start_subtime))
        right_event_sharp = models["concentration_net"].module.predict(batch_data["event"]["right"])
        if is_save_onnx and start_collect_onnx:
            onnx_inputs = {
                "concentration_net": {
                    "left_img": batch_data["event"]["left"].detach().cpu().numpy(),
                    "right_img": batch_data["event"]["right"].detach().cpu().numpy(),
                },
            }
            torch.onnx.export(
                models['concentration_net'].module,
                (
                    batch_data["event"]["left"],
                    batch_data["event"]["right"],
                ),
                os.path.join(save_root, "concentration_net.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["left_right_inputs",],
                output_names=["left_right_sharps",],
            )
            

        imageHeight, imageWidth = batch_data["event"]["left"].shape[-2:]
        batch_img_metas = {"h": imageHeight, "w": imageWidth}
        num_classes = models["objdet_head"].module.config["num_classes"]

        # ---------- disp pred net ----------
        start_subtime = time.time()
        pred_disparity_pyramid = models["disp_head"].module.predict(left_event_sharp, right_event_sharp)
        print("disp_head costs: {} sec.".format(time.time() - start_subtime))
        if is_save_onnx and start_collect_onnx:
            onnx_inputs["disp_head"] = {
                "left_img": left_event_sharp.detach().cpu().numpy(),
                "right_img": right_event_sharp.detach().cpu().numpy(),
            }
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
                output_names=["preds"],
            )

        # ---------- objdet net ----------
        start_subtime = time.time()
        left_detections = models["objdet_head"].module.predict(batch_data["event"]["left"])
        print("objdet_head costs: {} sec.".format(time.time() - start_subtime))
        if is_save_onnx and start_collect_onnx:
            onnx_inputs["objdet_head"] = {
                "left_event_voxel": batch_data["event"]["left"].detach().cpu().numpy(),
            }
            import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()
            torch.onnx.export(
                models['objdet_head'].module,
                (
                    batch_data["event"]["left"],
                ),
                os.path.join(save_root, "objdet_head.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["left_event_voxel"],
                output_names=["preds0", "preds100", "preds101", "preds102", "preds11"],
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
            start_subtime = time.time()
            (
                batch_track_bboxes_priors,
                batch_corresponding_previousdet_ids,
                batch_track_refined_bboxes,
                batch_track_refined_scores,
                batch_track_predicted_keypts,
            ) = models["local_tracking_head"].module.predict(
                batch_data["event"]["left"],
                [previous_detections[..., :4]],
                torch.zeros((1, imageHeight, imageWidth), device=device, dtype=dtype),
                batch_img_metas,
            )
            print("local_tracking_head costs: {} sec.".format(time.time() - start_subtime))

            if is_save_onnx and start_collect_onnx:
                torch.onnx.export(
                    models['local_tracking_head'].module,
                    (
                        batch_data["event"]["left"],
                        [previous_detections[..., :4]],
                        torch.zeros((1, imageHeight, imageWidth), device=device, dtype=dtype),
                        [rpn_hypotheses.get_dict()["bboxes"] for rpn_hypotheses in batch_rpn_hypotheses],
                        [rpn_hypotheses.get_dict()["target_ids"] for rpn_hypotheses in batch_rpn_hypotheses],
                    ),
                    os.path.join(save_root, "local_tracking_head.onnx"),
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=["left_event_voxel", "left_bboxes", "disp_prior", "batch_hypotheses_bboxes", "batch_hypotheses_target_ids"],
                    output_names=["batch_sbboxes_priors", "batch_refined_bboxes", "batch_refined_scores", "batch_predicted_keypts",]
                )
                onnx_inputs["local_tracking_head"] = {
                    "left_event_voxel": batch_data["event"]["left"].detach().cpu().numpy(),
                    "left_bboxes": previous_detections[..., :4].detach().cpu().numpy(),
                    "batch_hypotheses_bboxes": batch_rpn_hypotheses[0].numpy().get_dict()["bboxes"],
                    "batch_hypotheses_target_ids": batch_rpn_hypotheses[0].numpy().get_dict()["target_ids"],
                }
            
            if batch_track_bboxes_priors[0] is not None:
                max_num_keypoints = models["local_tracking_head"].module.config["max_num_keypoints"]
                (
                    mask_track_nonbackground,
                    tracked_bboxes_nobkg,
                    tracked_scores,
                    tracked_keypts_nobkg,
                ) = models["local_tracking_head"].module.extract_inference_results(
                    batch_track_bboxes_priors[0].squeeze(0),
                    batch_track_refined_bboxes[0].view(-1, num_classes, 4),
                    batch_track_refined_scores[0],
                    batch_track_predicted_keypts[0].view(-1, num_classes, max_num_keypoints * 3),
                )
                # if batch_data['end_timestamp'][0] == '2821053':
                #     import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()
                # logger.critical("batch_data['end_timestamp']: %s", batch_data['end_timestamp'][0])
                if mask_track_nonbackground is not None and mask_track_nonbackground.sum() > 0:
                    corresponding_previousdets_indices = batch_corresponding_previousdet_ids[0][mask_track_nonbackground]
                    corresponding_previousdets = previous_detections[corresponding_previousdets_indices]
                    tracked_class_labels = corresponding_previousdets[:, 8].unsqueeze(-1)
                    tracked_confidences = corresponding_previousdets[:, 9].unsqueeze(-1)
                    # filter these tracked bboxes with tracking threshold (same magitude as stereo threshold)
                    score_threshold = models["local_tracking_head"].module.config["right_confidence_threshold_inference"]
                    # mask_good_tracked = torch.logical_and(tracked_scores.view(-1) >= score_threshold, tracked_class_labels.squeeze() != 0)  # Hack:  do not track the class:0
                    mask_good_tracked = tracked_scores.view(-1) >= score_threshold
                    # filter by width diff change
                    width_diff = torch.abs((tracked_bboxes_nobkg[:, 6] - tracked_bboxes_nobkg[:, 4]) - (tracked_bboxes_nobkg[:, 2] - tracked_bboxes_nobkg[:, 0])) / (tracked_bboxes_nobkg[:, 2] - tracked_bboxes_nobkg[:, 0])
                    mask_good_tracked = torch.logical_and(
                        mask_good_tracked,
                        width_diff < (models["local_tracking_head"].module.config["left_right_width_diff_threshold"]- 1),
                    )
                    if mask_good_tracked.sum().item() > 0:
                        tracked_bboxes_nobkg = tracked_bboxes_nobkg[mask_good_tracked]
                        tracked_keypts_nobkg = tracked_keypts_nobkg[mask_good_tracked]
                        tracked_class_labels = tracked_class_labels[mask_good_tracked]
                        tracked_confidences = tracked_confidences[mask_good_tracked]
                        corresponding_previousdets = corresponding_previousdets[mask_good_tracked]
                        corresponding_previousdets[:, :4] = tracked_bboxes_nobkg[:, 4:]
                        corresponding_previousdets[:, 11:(11 + max_num_keypoints * 3)] = tracked_keypts_nobkg.view(tracked_keypts_nobkg.shape[0], -1)

        refined_sbboxes_nobkg = None
        if left_bboxesClsKeypts_nmsed_topked[0].shape[0] > 0:
            # ---------- stereo detection head ----------
            left_bboxes_nmsed_topked = [one_batch[..., :4] for one_batch in left_bboxesClsKeypts_nmsed_topked]
            left_confidences_nmsed_topked = torch.max(left_bboxesClsKeypts_nmsed_topked[0][:, 4:(4 + num_classes)], dim=-1)[0].unsqueeze(-1)
            if mask_track_nonbackground is not None and mask_track_nonbackground.sum() > 0:
                left_bboxes_nmsed_topked[0] = torch.concat([left_bboxes_nmsed_topked[0], tracked_bboxes_nobkg[:, 4:]], dim=0)
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

            start_subtime = time.time()
            (
                batch_sbboxes_priors,
                batch_corresponding_leftdet_ids,
                batch_refined_right_bboxes,
                batch_refined_right_scores,
                batch_predicted_right_keypts,
                rpn_cls_scores,
                rpn_bbox_preds,
                batch_rpn_hypotheses,
            ) = models["stereo_detection_head"].module.predict(
                batch_data["event"]["right"],
                left_bboxes_nmsed_topked,
                pred_disparity_pyramid[-1],
                batch_img_metas,
            )
            print("stereo_detection_head costs: {} sec.".format(time.time() - start_subtime))
            if is_save_onnx and start_collect_onnx:
                torch.onnx.export(
                    models['stereo_detection_head'].module,
                    (
                        batch_data["event"]["right"],
                        left_bboxes_nmsed_topked,
                        pred_disparity_pyramid[-1],
                        [rpn_hypotheses.get_dict()["bboxes"] for rpn_hypotheses in batch_rpn_hypotheses],
                        [rpn_hypotheses.get_dict()["target_ids"] for rpn_hypotheses in batch_rpn_hypotheses],
                    ),
                    os.path.join(save_root, "stereo_detection_head.onnx"),
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=["right_event_voxel", "left_bboxes", "disp_prior", "batch_hypotheses_bboxes", "batch_hypotheses_target_ids"],
                    output_names=["batch_sbboxes_priors", "batch_refined_right_bboxes", "batch_refined_right_scores", "batch_predicted_right_keypts", "rpn_cls_scores", "rpn_bbox_preds"],
                )
                onnx_inputs["stereo_detection_head"] = {
                    "right_event_voxel": batch_data["event"]["right"].detach().cpu().numpy(),
                    "left_bboxes": left_bboxes_nmsed_topked[0].detach().cpu().numpy(),
                    "batch_hypotheses_bboxes": batch_rpn_hypotheses[0].numpy().get_dict()["bboxes"],
                    "batch_hypotheses_target_ids": batch_rpn_hypotheses[0].numpy().get_dict()["target_ids"],
                }
                torch.save(onnx_inputs, "onnx_inputs.pth")

            assert left_event_sharp.shape[0] == 1  # batch size should be 1
            if batch_sbboxes_priors[0] is not None:
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
                    batch_predicted_right_keypts[0].view(-1, num_classes, models["stereo_detection_head"].module.config["max_num_keypoints"] * 3),
                )

        logger.info("one infer time: {} sec.".format(time.time() - starttime))
        infer_time.append(time.time() - starttime)
        if is_save_onnx and (left_bboxesClsKeypts_nmsed_topked[0].shape[0] > 0) and start_collect_onnx:
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
                max_num_keypoints = models["stereo_detection_head"].module.config["max_num_keypoints"]
                left_keypts_final = torch.concat([
                    left_keypts_final,
                    leftdets_tracked[:, 11:(11 + max_num_keypoints * 3)].view(-1, max_num_keypoints * 3),
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
                iou_threshold=models["objdet_head"].module.config["confidence_threshold_inference"],
            )
            raw_preds = raw_preds[left_keep_indices]

            preds = FilterBadDetections(
                raw_preds,
                imageHeight=batch_data["image_metadata"]["h_recti"],
                imageWidth=batch_data["image_metadata"]["w_recti"],
                margin=models["stereo_detection_head"].module.config["invalid_distance_from_image_border"],
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
                # mask redundant keypts
                max_num_keypoints = models["stereo_detection_head"].module.config["max_num_keypoints"]
                class_num_keypoints = {
                    0: 2,
                    1: 4,
                }
                for indexPred in range(preds.shape[0]):
                    class_label = int(preds[indexPred, 8].item())
                    num_keypoints = class_num_keypoints.get(class_label, 0)
                    preds[indexPred, (11 + num_keypoints * 3):(11 + max_num_keypoints * 3)] = -1
                    preds[indexPred, (11 + max_num_keypoints * 3 + num_keypoints * 3):(11 + max_num_keypoints * 3 * 2)] = -1

                prediction_dict = {
                    "objdet": [preds],
                    "concentrate": {
                        "left": left_event_sharp,
                        "right": right_event_sharp,
                    },
                    "ts": batch_data["end_timestamp"][0],
                    "disp": cv2.cvtColor(pred_disparity_pyramid[-1].detach().cpu().numpy().astype('uint8')[0], cv2.COLOR_GRAY2BGR),
                }
                stereo_visz = SaveTestResultsAndVisualize(
                    prediction_dict,
                    indexBatch,
                    batch_data["end_timestamp"][0],
                    sequence_name,
                    save_root,
                    batch_data["image_metadata"],
                )
                num_final_detections[batch_data["end_timestamp"][0]] = preds.shape[0]

                if is_save_onnx and not start_collect_onnx:
                    if preds.shape[0] >= 1:
                        start_collect_onnx = True
                        batch_data_for_onnx = copy.deepcopy(batch_data)
                        print("start to collect onnx inputs and outputs...")
                # # -------------- debug code --------------
                # os.makedirs("/root/data/debug_test/", exist_ok=True)
                # h, w = stereo_visz[0].shape[:2]
                # h = h // 2
                # cv2.imwrite("/root/data/debug_test/" + str(previous_prediction_dict['ts']) + ".png", numpy.vstack([previous_prediction_dict['disp'][:h, :w], stereo_visz[0]]))
                # # -------------- debug code --------------
            else:
                prediction_dict = None
                logger.error("batch {} has no valid detections.".format(indexBatch))
        else:
            prediction_dict = None
            logger.error("batch {} has no valid detections.".format(indexBatch))

        # update evals_input
        if batch_data.get('gt_labels', None) is not None:
            if prediction_dict is None:
                evals_input["FN"].append(
                    batch_data['gt_labels']['objdet'][0]['bboxes'].shape[0]
                )
                evals_input["TP"].append(0)
                evals_input["FP"].append(0)
                evals_input["GT"].append(batch_data['gt_labels']['objdet'][0]['bboxes'].shape[0])
            else:
                TP, FP, FN = evaluate_results_with_gt(
                    prediction_dict['objdet'][0][:, :8],
                    prediction_dict['objdet'][0][:, 8],
                    batch_data['gt_labels']['objdet'][0]['bboxes'][:, [0, 1, 2, 3, 4, 1, 5, 3]],
                    batch_data['gt_labels']['objdet'][0]['labels'],
                    iou_threshold,
                )
                evals_input["TP"].append(TP)
                evals_input["FP"].append(FP)
                evals_input["FN"].append(FN)
                evals_input["GT"].append(batch_data['gt_labels']['objdet'][0]['bboxes'].shape[0])

        # if no detection, save an empty stereo image
        if prediction_dict is None:
            left_image = left_event_sharp.cpu().squeeze().numpy()
            left_image = left_image - left_image.min()
            left_image = (left_image / left_image.max() * 255.0).astype('uint8')
            right_image = right_event_sharp.cpu().squeeze().numpy()
            right_image = right_image - right_image.min()
            right_image = (right_image / right_image.max() * 255.0).astype('uint8')
            stereo_image = numpy.hstack([left_image[:418, :578], right_image[:418, :578]])
            ts = batch_data["end_timestamp"][0]
            cv2.imwrite(
                os.path.join(save_root, "inference", "det_visz", sequence_name, f"{ts:s}.png"),
                stereo_image,
            )
            cv2.imwrite(
                os.path.join(save_root, "inference", "left", sequence_name, f"{ts:s}.png"),
                left_image[:418, :578],
            )

        if batch_data["end_timestamp"][0] not in num_final_detections:
            num_final_detections[batch_data["end_timestamp"][0]] = 0
        pbar.update(1)
    print("average infer time: {} sec.".format(sum(infer_time) / len(infer_time)))
    print("mean detections: {}".format(numpy.array(list(num_final_detections.values())).mean()))
    if evals_input["GT"]:
        # dump eval_results if gt available
        with open(os.path.join(save_root, f"{sequence_name}_eval_counts.pkl"), "wb") as f:
            pickle.dump(evals_input, f)
# import os
# import pickle
# with open(os.path.join('/root/code/docker_pytorch_trainnn/experiments/traffic_signs/', "seq0_eval_counts.pkl"), "rb") as f:
#     seq0_evals_input = pickle.load(f)
# with open(os.path.join('/root/code/docker_pytorch_trainnn/experiments/traffic_signs/', "seq1_eval_counts.pkl"), "rb") as f:
#     seq1_evals_input = pickle.load(f)
# seq0_evals_input
# seq0_evals_input.keys()
# TPs = seq0_evals_input['TP'] + seq1_evals_input['TP']
# FPs = seq0_evals_input['FP'] + seq1_evals_input['FP']
# FNs = seq0_evals_input['FN'] + seq1_evals_input['FN']
# TTP = sum(TPs)
# TFP = sum(FPs)
# TFN = sum(FNs)
# TTP
# TFP
# TFN
# precision = TTP / (TTP + TFP)
# recall = TTP / (TTP + TFN)

    with open(os.path.join(save_root, f"{sequence_name}_num_final_detections.pkl"), "wb") as f:
        pickle.dump(num_final_detections, f)
    pbar.close()
    return


def evaluate_mAP(
    models,
    data_loader,
    sequence_name,
    save_root,
    is_save_onnx = False
):
    for model in models.values():
        model.module.eval()

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    prediction_dict = None
    start_collect_onnx = False  # start collecting when enough detections emerged.
    batch_data_for_onnx = None

    max_detection_threshold = 100  # from torchmetrics
    iou_threshold = 0.1  # initial iou threshold

    pred_bboxes = {
        "left": [],
        "update": [],
        "stereo": [],
    }
    pred_scores = copy.deepcopy(pred_bboxes)
    pred_labels = copy.deepcopy(pred_bboxes)
    target_bboxes = copy.deepcopy(pred_bboxes)
    target_labels = copy.deepcopy(pred_bboxes)
    for indexBatch in range(len(data_loader.dataset)):
        batch_data = batch_to_cuda(next(data_iter))
        if not batch_data['event'] or batch_data['event'].get('left') is None:
            pbar.update(1)
            logger.warning("batch {} has no event data.".format(indexBatch))
            continue
        
        if is_save_onnx and start_collect_onnx and batch_data_for_onnx is not None:
            batch_data = batch_data_for_onnx
        
        starttime = time.time()
        # ---------- concentration net ----------
        start_subtime = time.time()
        left_event_sharp = models["concentration_net"].module.predict(batch_data["event"]["left"])
        print("concentration_net costs: {} sec.".format(time.time() - start_subtime))
        right_event_sharp = models["concentration_net"].module.predict(batch_data["event"]["right"])

        imageHeight, imageWidth = batch_data["event"]["left"].shape[-2:]
        batch_img_metas = {"h": imageHeight, "w": imageWidth}
        num_classes = models["objdet_head"].module.config["num_classes"]

        # ---------- disp pred net ----------
        start_subtime = time.time()
        pred_disparity_pyramid = models["disp_head"].module.predict(left_event_sharp, right_event_sharp)
        print("disp_head costs: {} sec.".format(time.time() - start_subtime))

        # ---------- objdet net ----------
        start_subtime = time.time()
        left_detections = models["objdet_head"].module.predict(batch_data["event"]["left"])
        print("objdet_head costs: {} sec.".format(time.time() - start_subtime))

        left_detections_multilevels_detachcopy = DetachCopyNested(left_detections)
        left_bboxesClsKeypts_nmsed_topked, nms_topk_mask = non_max_suppression(
            left_detections_multilevels_detachcopy,
            conf_thres=0.0,  # for evaluation, do not filter with confidence here.
            iou_thres=iou_threshold,
            labels=[],
            nc=models["objdet_head"].module.config["num_classes"],
            multi_label=False,
            agnostic=False,
            max_det=max_detection_threshold,
            end2end=False,
        )
        pred_bboxes['left'].append(
            left_bboxesClsKeypts_nmsed_topked[0][:, :4]
        )
        pred_scores['left'].append(
            torch.max(left_bboxesClsKeypts_nmsed_topked[0][:, 4:(4 + num_classes)], dim=-1)[0]
        )
        left_labels = torch.argmax(left_bboxesClsKeypts_nmsed_topked[0][:, 4:(4 + num_classes)], dim=-1)
        pred_labels['left'].append(
            left_labels
        )
        target_bboxes['left'].append(
            batch_data['gt_labels']['objdet'][0]['bboxes'][:, :4]
        )
        target_labels['left'].append(
            batch_data['gt_labels']['objdet'][0]['labels'].to(dtype=torch.int64)
        )

    preds = [
        dict(
        boxes=bboxes,
        scores=scores,
        labels=labels,
        ) for bboxes, scores, labels in zip(pred_bboxes['left'], pred_scores['left'], pred_labels['left'])
    ]
    target = [
        dict(
        boxes=bboxes,
        labels=labels,
        ) for bboxes, labels in zip(target_bboxes['left'], target_labels['left'])
    ]
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, target)
    print("left network:")
    print(metric.compute())

    evals_input = {
        'pred_bboxes': pred_bboxes['left'],
        'pred_scores': pred_scores['left'],
        'pred_labels': pred_labels['left'],
        'target_bboxes': target_bboxes['left'],
        'target_labels': target_labels['left'],
    }
    with open(os.path.join(save_root, f"{sequence_name}_mAP_inputs.pkl"), "wb") as f:
        pickle.dump(evals_input, f)
