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
    )

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)

    for indexBatch in range(len(data_loader)):
        batch_data = batch_to_cuda(next(data_iter))

        mask = batch_data["disparity"] > 0
        if not mask.any():
            continue

        pred, lossDict = model(
            left_event=batch_data["event"]["left"],
            right_event=batch_data["event"]["right"],
            gt_disparity=batch_data["disparity"],
        )

        optimizer.zero_grad()
        loss = 0.
        for key, value in lossDict.items():
            loss += value
            log_dict[key].update(lossDict[key].item(), pred.size(0))
        loss.backward()
        optimizer.step()

        log_dict["Loss"].update(loss.item(), pred.size(0))
        log_dict["EPE"].update(pred, batch_data["disparity"], mask)
        log_dict["1PE"].update(pred, batch_data["disparity"], mask)
        log_dict["2PE"].update(pred, batch_data["disparity"], mask)
        log_dict["RMSE"].update(pred, batch_data["disparity"], mask)

        pbar.update(1)
    pbar.close()

    return log_dict


@torch.no_grad()
def valid(model, data_loader, is_distributed=False, world_size=1, logger=None):
    model.eval()

    log_dict = OrderedDict(
        [
            ("BestIndex", AverageMeter(string_format="%6.3lf")),
            ("EPE", EndPointError(average_by="image", string_format="%6.3lf")),
            ("1PE", NPixelError(n=1, average_by="image", string_format="%6.3lf")),
            ("2PE", NPixelError(n=2, average_by="image", string_format="%6.3lf")),
            ("RMSE", RootMeanSquareError(average_by="image", string_format="%6.3lf")),
        ]
    )

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader)):
        batch_data = batch_to_cuda(next(data_iter))

        mask = batch_data["disparity"] > 0
        if not mask.any():
            continue

        pred, lossDict = model(
            left_event=batch_data["event"]["left"],
            right_event=batch_data["event"]["right"],
            gt_disparity=batch_data["disparity"],
        )

        loss = 0.
        for key, value in lossDict.items():
            loss += value
            log_dict[key].update(lossDict[key].item(), pred.size(0))
        EPE_thisBatch = log_dict["EPE"].update(pred, batch_data["disparity"], mask)
        log_dict["BestIndex"].update(loss.item(), batch_data["disparity"].size(0))
        log_dict["1PE"].update(pred, batch_data["disparity"], mask)
        log_dict["2PE"].update(pred, batch_data["disparity"], mask)
        log_dict["RMSE"].update(pred, batch_data["disparity"], mask)

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


def batch_to_cuda(batch_data):
    def _batch_to_cuda(batch_data):
        if isinstance(batch_data, dict):
            for key in batch_data.keys():
                batch_data[key] = _batch_to_cuda(batch_data[key])
        elif isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.cuda()
        else:
            raise NotImplementedError

        return batch_data

    for domain in ["event"]:
        if domain not in batch_data.keys():
            batch_data[domain] = {}
        for location in ["left", "right"]:
            if location in batch_data[domain].keys():
                batch_data[domain][location] = _batch_to_cuda(
                    batch_data[domain][location]
                )
            else:
                batch_data[domain][location] = None
    if "disparity" in batch_data.keys() and batch_data["disparity"] is not None:
        batch_data["disparity"] = batch_data["disparity"].cuda()

    return batch_data
