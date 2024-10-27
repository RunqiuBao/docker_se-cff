import os.path
import numpy
import torch
import torch.distributed as dist
import cv2
import time
import torchvision

from tqdm import tqdm
from collections import OrderedDict

from utils.metrics import AverageMeter, EndPointError, NPixelError, RootMeanSquareError
from utils import visualizer
from ..methods.visz_utils import DrawResultBboxesAndKeyptsOnStereoEventFrame, RenderImageWithBboxes


def train(
    model,
    data_loader,
    optimizer,
    scaler,
    ema,
    clip_max_norm=None,
    is_distributed=False,
    world_size=1,
    epoch=None
):
    model.train()

    log_dict = OrderedDict(
        [
            ("Loss", AverageMeter(string_format="%6.3lf")),
            ("l1_loss", AverageMeter(string_format="%6.3lf")),
            ("warp_loss", AverageMeter(string_format="%6.3lf")),
            ("EPE", EndPointError(average_by="image", string_format="%6.3lf")),
            ("1PE", NPixelError(n=1, average_by="image", string_format="%6.3lf")),
            ("2PE", NPixelError(n=2, average_by="image", string_format="%6.3lf")),
            ("RMSE", RootMeanSquareError(average_by="image", string_format="%6.3lf")),
        ]
    ) if (hasattr(model.module, 'is_freeze_disp') and not model.module.is_freeze_disp) else OrderedDict(
        [
            ("Loss", AverageMeter(string_format="%6.3lf")),
            ("loss_cls", AverageMeter(string_format="%6.3lf")),
            ("loss_bbox", AverageMeter(string_format="%6.3lf")),
            ("loss_rbbox", AverageMeter(string_format="%6.3lf")),
            ("loss_rscore", AverageMeter(string_format="%6.3lf")),
            ("loss_obj", AverageMeter(string_format="%6.3lf")),
            ("loss_keypt1", AverageMeter(string_format="%6.3lf")),
            ("loss_keypt2", AverageMeter(string_format="%6.3lf")),
            ("loss_pmap", AverageMeter(string_format="%6.3lf"))
        ]
    )

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)

    for indexBatch in range(len(data_loader)):
        batch_data = batch_to_cuda(next(data_iter))        
        
        # print("max disp: {}".format(batch_data["gt_labels"]["disparity"].max()))
        if hasattr(model.module, 'is_freeze_disp') and not model.module.is_freeze_disp:
            mask = batch_data["gt_labels"]["disparity"] > 0
            if not mask.any():
                continue

        for key, suboptimizer in optimizer.items():
            suboptimizer.zero_grad()

        global_step_info = dict(epoch=epoch, indexBatch=indexBatch, lengthDataLoader=len(data_loader))
        if scaler is not None:
            with torch.autocast(device_type="cuda", cache_enabled=True):
                pred, lossDict = model(
                    left_event=batch_data["event"]["left"],
                    right_event=batch_data["event"]["right"],
                    gt_labels=batch_data["gt_labels"],
                    batch_img_metas=batch_data["image_metadata"],
                    global_step_info=global_step_info
                )

            loss = 0.
            for key, value in lossDict.items():
                loss += value
                if key in log_dict:
                    log_dict[key].update(lossDict[key].item(), data_loader.batch_size)
            scaler.scale(loss).backward()

            if clip_max_norm > 0:
                for key, suboptimizer in optimizer.items():
                    scaler.unscale_(suboptimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

            for key, suboptimizer in optimizer.items():
                scaler.step(suboptimizer)

            scaler.update()
        else:
            pred, lossDict = model(
                left_event=batch_data["event"]["left"],
                right_event=batch_data["event"]["right"],
                gt_labels=batch_data["gt_labels"],
                batch_img_metas=batch_data["image_metadata"],
                global_step_info=global_step_info
            )
            loss = 0.
            for key, value in lossDict.items():
                loss += value
                if key in log_dict:
                    log_dict[key].update(lossDict[key].item(), data_loader.batch_size)
            loss.backward()
            for key, suboptimizer in optimizer.items():
                suboptimizer.step()

        if ema is not None:
            ema.update(model)

        log_dict["Loss"].update(loss.item(), data_loader.batch_size)
        if hasattr(model.module, 'is_freeze_disp') and not model.module.is_freeze_disp:
            log_dict["EPE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["1PE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["2PE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["RMSE"].update(pred['disparity'], batch_data["disparity"], mask)
        else:
            for key, value in lossDict.items():
                log_dict[key].update(value, data_loader.batch_size)

        pbar.update(1)
        torch.cuda.synchronize()

    pbar.close()

    return log_dict


@torch.no_grad()
def valid(model, data_loader, ema=None, is_distributed=False, world_size=1, logger=None, epoch=None):
    """
    Args:
        ...
        ema: Exponential Moving Average. Smoothing between epoches.
    """
    model.eval()

    log_dict = OrderedDict(
        [
            ("BestIndex", AverageMeter(string_format="%6.3lf")),
            ("Loss", AverageMeter(string_format="%6.3lf")),
            ("l1_loss", AverageMeter(string_format="%6.3lf")),
            ("warp_loss", AverageMeter(string_format="%6.3lf")),
            ("EPE", EndPointError(average_by="image", string_format="%6.3lf")),
            ("1PE", NPixelError(n=1, average_by="image", string_format="%6.3lf")),
            ("2PE", NPixelError(n=2, average_by="image", string_format="%6.3lf")),
            ("RMSE", RootMeanSquareError(average_by="image", string_format="%6.3lf")),
        ]
    ) if (hasattr(model, 'is_freeze_disp') and not model.is_freeze_disp) else OrderedDict(
        [
            ("BestIndex", AverageMeter(string_format="%6.3lf")),
            ("Loss", AverageMeter(string_format="%6.3lf")),
            ("loss_cls", AverageMeter(string_format="%6.3lf")),
            ("loss_bbox", AverageMeter(string_format="%6.3lf")),
            ("loss_rbbox", AverageMeter(string_format="%6.3lf")),
            ("loss_rscore", AverageMeter(string_format="%6.3lf")),
            ("loss_obj", AverageMeter(string_format="%6.3lf")),
            ("loss_keypt1", AverageMeter(string_format="%6.3lf")),
            ("loss_keypt2", AverageMeter(string_format="%6.3lf")),
            ("loss_pmap", AverageMeter(string_format="%6.3lf"))
        ]
    )

    if logger is not None:
        pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader)):
        batch_data = batch_to_cuda(next(data_iter))

        if hasattr(model, 'is_freeze_disp') and not model.is_freeze_disp:
            mask = batch_data["gt_labels"]["disparity"] > 0
            if not mask.any():
                continue

        global_step_info = dict(epoch=epoch, indexBatch=indexBatch, lengthDataLoader=len(data_loader))

        pred, lossDict = model(
            left_event=batch_data["event"]["left"],
            right_event=batch_data["event"]["right"],
            gt_labels=batch_data["gt_labels"],
            batch_img_metas=batch_data["image_metadata"],
            global_step_info=global_step_info
        )

        loss = 0.
        for key, value in lossDict.items():
            loss += value
            if key in log_dict:
                log_dict[key].update(lossDict[key].item(), data_loader.batch_size)

        log_dict["BestIndex"].update(loss.item(), data_loader.batch_size)
        if hasattr(model, 'is_freeze_disp') and not model.is_freeze_disp:
            log_dict["EPE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["1PE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["2PE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["RMSE"].update(pred['disparity'], batch_data["disparity"], mask)
        else:
            for key, value in lossDict.items():
                log_dict[key].update(value, data_loader.batch_size)

        if logger is not None:
            pbar.update(1)

        torch.cuda.synchronize()

    if logger is not None:
        pbar.close()

    return log_dict


@torch.no_grad()
def test(
    model,
    data_loader,
    sequence_name,
    save_root
):
    model.eval()

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader.dataset)):            
        batch_data = batch_to_cuda(next(data_iter))
        starttime = time.time()
        pred, _ = model(
            left_event=batch_data["event"]["left"],
            right_event=batch_data["event"]["right"],
            gt_labels=None,
            batch_img_metas=batch_data["image_metadata"]
        )
        print("one infer time: {}".format(time.time() - starttime))
        
        SaveTestResults(
            pred,
            indexBatch,
            batch_data["end_timestamp"].item(),
            sequence_name,
            save_root,
            batch_data["image_metadata"],
            model.module.__class__.__name__
        )
        pbar.update(1)
    pbar.close()
    return


def batch_to_cuda(batch_data, dtype=torch.float32):
    def _batch_to_cuda(batch_data, dtype):
        if isinstance(batch_data, dict):
            for key in batch_data.keys():
                batch_data[key] = _batch_to_cuda(batch_data[key], dtype=dtype)
        elif isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.to(dtype).cuda()
        else:
            raise NotImplementedError

        return batch_data

    for domain in ["event"]:
        if domain not in batch_data.keys():
            batch_data[domain] = {}
        for location in ["left", "right"]:
            if location in batch_data[domain].keys():
                batch_data[domain][location] = _batch_to_cuda(
                    batch_data[domain][location], dtype=dtype
                )
            else:
                batch_data[domain][location] = None
    if "disparity" in batch_data.keys() and batch_data["disparity"] is not None:
        batch_data["disparity"] = batch_data["disparity"].to(dtype).cuda()

    if "gt_labels" in batch_data:
        for side in ["left", "right"]:
            for i in range(len(batch_data["gt_labels"]['objdet'][side])):
                for key in batch_data["gt_labels"]['objdet'][side][i].keys():   
                    if isinstance(batch_data["gt_labels"]["objdet"][side][i][key], torch.Tensor):             
                        batch_data["gt_labels"]["objdet"][side][i][key] = batch_data["gt_labels"]["objdet"][side][i][key].cuda()    
                    elif isinstance(batch_data["gt_labels"]["objdet"][side][i][key], dict):
                        for subkey in batch_data["gt_labels"]["objdet"][side][i][key]:
                            batch_data["gt_labels"]["objdet"][side][i][key][subkey] = batch_data["gt_labels"]["objdet"][side][i][key][subkey].cuda()
                    elif isinstance(batch_data["gt_labels"]["objdet"][side][i][key], list):
                        for ii in range(len(batch_data["gt_labels"]["objdet"][side][i][key])):
                            batch_data["gt_labels"]["objdet"][side][i][key][ii] = batch_data["gt_labels"]["objdet"][side][i][key][ii].cuda()

                # batch_data["gt_labels"]["objdet"][i]["bboxes"] = batch_data["gt_labels"]["objdet"][i]["bboxes"].to(dtype).cuda()
                # batch_data["gt_labels"]["objdet"][i]["labels"] = batch_data["gt_labels"]["objdet"][i]["labels"].to(dtype).cuda()
                # batch_data["gt_labels"]["objdet"][i]["keypt1_masks"] = batch_data["gt_labels"]["objdet"][i]["keypt1_masks"].to(dtype).cuda()
                # batch_data["gt_labels"]["objdet"][i]["keypt2_masks"] = batch_data["gt_labels"]["objdet"][i]["keypt2_masks"].to(dtype).cuda()

    return batch_data


def SaveTestResults(pred: dict, indexBatch: int, timestamp: int, sequence_name: str, save_root: str, img_metas: dict, model_name: str):    
    if model_name == "EventStereoSegmentationNetwork":
        SaveTestResultsForBinpickingTarget(pred, indexBatch, timestamp, sequence_name, save_root, img_metas, model_name)
    elif model_name == "EventPickTargetPredictionNetwork":
        SaveTestResultsForBinpickingTarget(pred, indexBatch, timestamp, sequence_name, save_root, img_metas, model_name)
    elif model_name == "EventStereoObjectDetectionNetwork":
        SaveTestResultsAndVisualize(pred, indexBatch, timestamp, sequence_name, save_root, img_metas)
    
    else:
        raise NotImplementedError


def SaveTestResultsAndVisualize(pred: dict, indexBatch: int, timestamp: int, sequence_name: str, save_root: str, img_metas: dict):
    """
    for EventStereoObjectDetectionNetwork.
    """
    # save detection results in txt, frame by frame.
    path_det_results_folder = os.path.join(save_root, "inference", "detections", sequence_name)
    path_tsfile = os.path.join(save_root, "inference", "detections", sequence_name, "timestamps.txt")
    os.makedirs(path_det_results_folder, exist_ok=True)
    ts_openmode = "a" if os.path.isfile(path_tsfile) else "w"
    with open(path_tsfile, ts_openmode) as tsfile:
        tsfile.write(str(timestamp) + "\n")
    detresults_openmode = "w"
    batch_size = len(pred['objdet'])
    for indexInBatch, detection in enumerate(pred['objdet']):
        with open(os.path.join(path_det_results_folder, str(indexBatch * batch_size + indexInBatch).zfill(6) + ".txt"), detresults_openmode) as detresult_file:
            for indexDet in range(detection.shape[0]):
                oneDet = numpy.array2string(detection[indexDet].cpu().numpy(), separator=" ", max_line_width=numpy.inf, formatter={'float_kind':lambda x: "%.4f" % x})[1:-1]
                detresult_file.write(oneDet + "\n")

    # save detection visualization results, frame by frame. Concate left and right horizontally.
    path_det_visz_folder = os.path.join(save_root, "inference", "det_visz", sequence_name)
    path_concentrate_left_folder = os.path.join(save_root, "inference", "left", sequence_name)
    path_concentrate_right_folder = os.path.join(save_root, "inference", "right", sequence_name)
    os.makedirs(path_det_visz_folder, exist_ok=True)
    os.makedirs(path_concentrate_left_folder, exist_ok=True)
    os.makedirs(path_concentrate_right_folder, exist_ok=True)
    imgHeight, imgWidth = img_metas['h_cam'], img_metas['w_cam']
    for indexInBatch, detection in enumerate(pred['objdet']):
        left_bboxes = detection[:, 1:5].cpu().numpy()
        left_bboxes[:, [0, 2]] *= imgWidth
        left_bboxes[:, [1, 3]] *= imgHeight
        tl_x = left_bboxes[:, 0] - left_bboxes[:, 2] / 2
        tl_y = left_bboxes[:, 1] - left_bboxes[:, 3] / 2
        br_x = left_bboxes[:, 0] + left_bboxes[:, 2] / 2
        br_y = left_bboxes[:, 1] + left_bboxes[:, 3] / 2
        left_bboxes = numpy.stack([tl_x, tl_y, br_x, br_y], axis=1)
        right_bboxes = detection[:, [5, 7]].cpu().numpy() * imgWidth
        tl_x_r = right_bboxes[:, 0] - right_bboxes[:, 1] / 2
        br_x_r = right_bboxes[:, 0] + right_bboxes[:, 1] / 2
        right_bboxes = numpy.stack([tl_x_r, br_x_r], axis=1)
        sbboxes = numpy.concatenate([left_bboxes, right_bboxes], axis=-1)
        classes = detection[:, 0].cpu().numpy().astype('int')
        confidences = detection[:, -4].cpu().numpy()
        stereo_confidences = detection[:, -3].cpu().numpy()
        keypts1 = detection[:, -8:-6].cpu().numpy()
        keypts2 = detection[:, -6:-4].cpu().numpy()        
        visz_left, visz_right = DrawResultBboxesAndKeyptsOnStereoEventFrame(
            pred['concentrate']['left'].squeeze().cpu().numpy()[:img_metas['h_cam'], :img_metas['w_cam']],
            pred['concentrate']['right'].squeeze().cpu().numpy()[:img_metas['h_cam'], :img_metas["w_cam"]],
            sbboxes,
            classes,
            confidences,
            keypts1,
            keypts2,
            stereo_confidences=stereo_confidences)
        visz = numpy.concatenate([visz_left, visz_right], axis=-2)
        visz = cv2.cvtColor(visz, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path_det_visz_folder, str(indexBatch * batch_size + indexInBatch).zfill(6) + ".png"), visz)
        left_concentrated = pred['concentrate']['left'].squeeze().cpu().numpy()[:img_metas['h_cam'], :img_metas['w_cam']]
        left_concentrated = left_concentrated - left_concentrated.min()
        left_concentrated = (left_concentrated * 255 / left_concentrated.max()).astype('uint8')
        right_concentrated = pred['concentrate']['right'].squeeze().cpu().numpy()[:img_metas['h_cam'], :img_metas['w_cam']]
        right_concentrated = right_concentrated - right_concentrated.min()
        right_concentrated = (right_concentrated * 255 / right_concentrated.max()).astype('uint8')
        cv2.imwrite(os.path.join(path_concentrate_left_folder, str(indexBatch * batch_size + indexInBatch).zfill(6) + ".png"), left_concentrated)
        cv2.imwrite(os.path.join(path_concentrate_right_folder, str(indexBatch * batch_size + indexInBatch).zfill(6) + ".png"), right_concentrated)
    return


def SaveTestResultsForBinpickingTarget(pred: dict, indexBatch: int, timestamp: int, sequence_name: str, save_root: str, img_metas: dict, model_name: str):
    if model_name == "EventStereoSegmentationNetwork":
        # save binpicking target predict results (semantic mask style), using alpha rendering.
        batch_size = pred["left_pmap"].shape[0]
        for indexInBatch in range(batch_size):
            path_det_visz_folder = os.path.join(save_root, "inference", "det_visz", sequence_name)
            os.makedirs(path_det_visz_folder, exist_ok=True)
            imgHeight, imgWidth = img_metas["h_cam"], img_metas["w_cam"]
            left_pmap = pred["left_pmap"].squeeze().cpu().numpy()[:imgHeight, :imgWidth]
            left_pmap -= left_pmap.min()
            left_pmap = (left_pmap * 255 / left_pmap.max()).astype("uint8")
            right_pmap = pred["right_pmap"].squeeze().cpu().numpy()[:imgHeight, :imgWidth]
            right_pmap -= right_pmap.min()
            right_pmap = (right_pmap * 255 / right_pmap.max()).astype("uint8")
            left_concentrated = pred['left_concentrate'].squeeze().cpu().numpy()[:imgHeight, :imgWidth]
            left_concentrated = numpy.abs(left_concentrated)
            left_concentrated = (left_concentrated * 255 / left_concentrated.max()).astype('uint8')
            right_concentrated = pred['right_concentrate'].squeeze().cpu().numpy()[:imgHeight, :imgWidth]
            right_concentrated = numpy.abs(right_concentrated)
            right_concentrated = (right_concentrated * 255 / right_concentrated.max()).astype('uint8')
            left_view = cv2.addWeighted(left_pmap, 0.5, left_concentrated, 0.5, 0)
            right_view = cv2.addWeighted(right_pmap, 0.5, right_concentrated, 0.5, 0)
            stereo_view = cv2.hconcat([left_view, right_view])
            cv2.imwrite(os.path.join(path_det_visz_folder, str(indexBatch * batch_size + indexInBatch).zfill(6) + ".png"), stereo_view)
    elif model_name == "EventPickTargetPredictionNetwork":
        # save binpicking target prediction restuls (object detection style).
        batch_size = pred["left_sharprepr"].shape[0]
        for indexInBatch in range(batch_size):
            path_det_visz_folder = os.path.join(save_root, "inference", "det_visz", sequence_name)
            os.makedirs(path_det_visz_folder, exist_ok=True)
            imgHeight, imgWidth = img_metas["h_cam"], img_metas["w_cam"]
            left_concentrated = pred["left_sharprepr"][indexInBatch]
            right_concentrated = pred["right_sharprepr"][indexInBatch]
            cropHeight, cropWidth = left_concentrated.shape[-2:]
            left_bboxes = torchvision.ops.box_convert(pred["left_detections"], in_fmt="cxcywh", out_fmt="xyxy".lower())
            left_bboxes[:, [0, 2]] *= cropWidth
            left_bboxes[:, [1, 3]] *= cropHeight
            right_bboxes = torchvision.ops.box_convert(pred["right_detections"], in_fmt="cxcywh", out_fmt="xyxy".lower())
            right_bboxes[:, [0, 2]] *= cropWidth
            right_bboxes[:, [1, 3]] *= cropHeight
            left_view_img = RenderImageWithBboxes(
                left_concentrated.squeeze(),
                {
                    "bboxes": left_bboxes,
                    "classes": pred["left_classes"]
                },
                is_output_torch=False
            )[0][:imgHeight, :imgWidth]            
            right_view_img = RenderImageWithBboxes(
                right_concentrated.squeeze(),
                {
                    "bboxes": right_bboxes,
                    "classes": pred["right_classes"]
                },
                is_output_torch=False
            )[0][:imgHeight, :imgWidth]
            stereo_view = cv2.hconcat([left_view_img, right_view_img])
            cv2.imwrite(
                os.path.join(path_det_visz_folder, str(indexBatch * batch_size + indexInBatch).zfill(6) + ".png"),
                stereo_view
            )                    
    else:
        raise NotImplementedError
