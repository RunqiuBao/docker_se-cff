import os.path
import numpy
import torch
import torch.distributed as dist
import cv2

from tqdm import tqdm
from collections import OrderedDict

from utils.metrics import AverageMeter, EndPointError, NPixelError, RootMeanSquareError
from utils import visualizer


def train(model, data_loader, optimizer, is_distributed=False, world_size=1):
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
    ) if not model.module.is_freeze_disp else OrderedDict(
        [
            ("Loss", AverageMeter(string_format="%6.3lf")),
            ("loss_cls", AverageMeter(string_format="%6.3lf")),
            ("loss_bbox", AverageMeter(string_format="%6.3lf")),
            ("loss_rbbox", AverageMeter(string_format="%6.3lf")),
            ("loss_rscore", AverageMeter(string_format="%6.3lf")),
            ("loss_obj", AverageMeter(string_format="%6.3lf")),
            ("loss_keypt1", AverageMeter(string_format="%6.3lf")),
            ("loss_keypt2", AverageMeter(string_format="%6.3lf"))
        ]
    )

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)

    for indexBatch in range(len(data_loader)):
        batch_data = batch_to_cuda(next(data_iter))
        # print("max disp: {}".format(batch_data["gt_labels"]["disparity"].max()))

        mask = batch_data["gt_labels"]["disparity"] > 0
        if not mask.any():
            continue

        pred, lossDict = model(
            left_event=batch_data["event"]["left"],
            right_event=batch_data["event"]["right"],
            gt_labels=batch_data["gt_labels"],
            batch_img_metas=batch_data["image_metadata"]
        )

        optimizer.zero_grad()
        loss = 0.
        for key, value in lossDict.items():
            loss += value
            if key in log_dict:
                log_dict[key].update(lossDict[key].item(), data_loader.batch_size)
        loss.backward()
        optimizer.step()

        log_dict["Loss"].update(loss.item(), data_loader.batch_size)
        if not model.module.is_freeze_disp:
            log_dict["EPE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["1PE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["2PE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["RMSE"].update(pred['disparity'], batch_data["disparity"], mask)
        else:
            for key, value in lossDict.items():
                log_dict[key].update(value, data_loader.batch_size)

        pbar.update(1)
    pbar.close()

    return log_dict


@torch.no_grad()
def valid(model, data_loader, is_distributed=False, world_size=1, logger=None):
    model.eval()

    log_dict = OrderedDict(
        [
            ("BestIndex", AverageMeter(string_format="%6.3lf")),
            ("l1_loss", AverageMeter(string_format="%6.3lf")),
            ("warp_loss", AverageMeter(string_format="%6.3lf")),
            ("EPE", EndPointError(average_by="image", string_format="%6.3lf")),
            ("1PE", NPixelError(n=1, average_by="image", string_format="%6.3lf")),
            ("2PE", NPixelError(n=2, average_by="image", string_format="%6.3lf")),
            ("RMSE", RootMeanSquareError(average_by="image", string_format="%6.3lf")),
        ]
    ) if not model.module.is_freeze_disp else OrderedDict(
        [
            ("BestIndex", AverageMeter(string_format="%6.3lf")),
            ("loss_cls", AverageMeter(string_format="%6.3lf")),
            ("loss_bbox", AverageMeter(string_format="%6.3lf")),
            ("loss_rbbox", AverageMeter(string_format="%6.3lf")),
            ("loss_rscore", AverageMeter(string_format="%6.3lf")),
            ("loss_obj", AverageMeter(string_format="%6.3lf")),
            ("loss_keypt1", AverageMeter(string_format="%6.3lf")),
            ("loss_keypt2", AverageMeter(string_format="%6.3lf"))
        ]
    )

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader)):
        batch_data = batch_to_cuda(next(data_iter))

        mask = batch_data["gt_labels"]["disparity"] > 0
        if not mask.any():
            continue

        pred, lossDict = model(
            left_event=batch_data["event"]["left"],
            right_event=batch_data["event"]["right"],
            gt_labels=batch_data["gt_labels"],
            batch_img_metas=batch_data["image_metadata"]
        )

        loss = 0.
        for key, value in lossDict.items():
            loss += value
            if key in log_dict:
                log_dict[key].update(lossDict[key].item(), data_loader.batch_size)

        log_dict["BestIndex"].update(loss.item(), data_loader.batch_size)
        if not model.module.is_freeze_disp:
            log_dict["EPE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["1PE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["2PE"].update(pred['disparity'], batch_data["disparity"], mask)
            log_dict["RMSE"].update(pred['disparity'], batch_data["disparity"], mask)
        else:
            for key, value in lossDict.items():
                log_dict[key].update(value, data_loader.batch_size)

        pbar.update(1)
    pbar.close()

    return log_dict


@torch.no_grad()
def test(model, data_loader):
    model.eval()
    pred_list = []

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader.dataset)):
        batch_data = batch_to_cuda(next(data_iter))

        pred, _ = model(
            left_event=batch_data["event"]["left"],
            right_event=batch_data["event"]["right"],
            gt_disparity=None,
        )

        for idx in range(pred.size(0)):
            width = data_loader.dataset.WIDTH
            height = data_loader.dataset.HEIGHT
            cur_pred = pred[idx, :height, :width].cpu()
            cur_pred_dict = {
                "file_name": str(batch_data["file_index"][idx].item()).zfill(6)
                + ".png",
                "pred": visualizer.tensor_to_disparity_image(cur_pred),
                "pred_magma": visualizer.tensor_to_disparity_magma_image(
                    cur_pred, vmax=100
                ),
            }
            pred_list.append(cur_pred_dict)
        pbar.update(1)
    pbar.close()

    return pred_list


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
        for i in range(len(batch_data["gt_labels"]['objdet'])):
            batch_data["gt_labels"]["objdet"][i]["bboxes"] = batch_data["gt_labels"]["objdet"][i]["bboxes"].to(dtype).cuda()
            batch_data["gt_labels"]["objdet"][i]["labels"] = batch_data["gt_labels"]["objdet"][i]["labels"].to(dtype).cuda()

    return batch_data
