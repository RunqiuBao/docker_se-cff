import os.path
import torch
import cv2
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

        if prediction_dict:
            # ---------- local tracking head ----------
            previous_detections = prediction_dict['objdet'][0]
            device, dtype = previous_detections.device, previous_detections.dtype
            batch_track_bboxes_priors, batch_track_refined_bboxes, batch_track_refined_scores, batch_track_predicted_keypts = models["local_tracking_head"].module.predict(
                batch_data["event"]["left"],
                [previous_detections[..., :4]],
                torch.zeros((1, 2, imageHeight, imageWidth), device=device, dtype=dtype),
                batch_img_metas,
                models["local_tracking_head"].module.config["bbox_move_anchor_ticks"],
                models["local_tracking_head"].module.config["bbox_expand_factor"]
            )
            (
                mask_nonbackground,
                refined_bboxes_nobkg,
                refined_scored_pred,
                predicted_keypts_nobkg
            ) = models["local_tracking_head"].module.extract_inference_results(
                batch_track_bboxes_priors[0].squeeze(0),
                batch_track_refined_bboxes[0].view(-1, num_classes, 4),
                batch_track_refined_scores[0],
                batch_track_predicted_keypts[0].view(-1, num_classes, models["local_tracking_head"].module.config["max_num_keypoints"] * 3)
            )
            import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()

        refined_sbboxes_nobkg = None
        if left_bboxesClsKeypts_nmsed_topked[0].shape[0] > 0:
            # ---------- stereo detection head ----------
            left_bboxes_nmsed_topked = [one_batch[..., :4] for one_batch in left_bboxesClsKeypts_nmsed_topked]
            batch_sbboxes_priors, batch_refined_right_bboxes, batch_refined_right_scores, batch_predicted_right_keypts = models["stereo_detection_head"].module.predict(
                batch_data["event"]["right"],
                left_bboxes_nmsed_topked,
                pred_disparity_pyramid[-1],
                batch_img_metas,
                models["stereo_detection_head"].module.config["bbox_expand_anchor_ticks"]
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
                    output_names=["batch_sbboxes_priors", "batch_refined_right_bboxes", "batch_refined_right_scores", "batch_predicted_right_keypts"]
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
            left_bboxes_final = left_bboxesClsKeypts_nmsed_topked[0][mask_nonbackground][:, 0:4]
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

            preds = FilterBadDetections(
                torch.concat([
                    left_bboxes_final,
                    torch.concat([
                        refined_sbboxes_nobkg[:, 4].view(-1, 1),
                        left_bboxes_final[:, 1].view(-1, 1),
                        refined_sbboxes_nobkg[:, 5].view(-1, 1),
                        left_bboxes_final[:, 3].view(-1, 1)
                    ], dim=-1),
                    torch.argmax(left_bboxesClsKeypts_nmsed_topked[0][:, 4:(4 + num_classes)], dim=-1).unsqueeze(-1)[mask_nonbackground],
                    torch.max(left_bboxesClsKeypts_nmsed_topked[0][:, 4:(4 + num_classes)], dim=-1)[0].unsqueeze(-1)[mask_nonbackground],
                    refined_right_scored_pred.view(-1, 1),
                    left_bboxesClsKeypts_nmsed_topked[0][:, (4 + models["objdet_head"].module.config["num_classes"]):][mask_nonbackground],
                    right_keypts_pred_nobkg.view(-1, models["stereo_detection_head"].module.config["max_num_keypoints"] * 3),
                ], dim=1),
                imageHeight=batch_data["image_metadata"]["h_cam"],
                imageWidth=batch_data["image_metadata"]["w_cam"],
                margin=4,
                right_confidence_threshold=models["stereo_detection_head"].module.config["right_confidence_threshold_inference"],
                left_right_confidence_diff=models["stereo_detection_head"].module.config["left_right_confidence_diff_inference"],
            )
            # preds = FilterIrregularBboxes(preds, hw_ratiorange_class0=[1.9, 3.15])
            preds = FilterTemporal(
                preds,
                previous_preds,
                iou_threshold_for_matching=0.4,
                iou_leftright_for_filtering=0.7,
                area_change_threshold=0.7
            )

            if preds is not None:
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
