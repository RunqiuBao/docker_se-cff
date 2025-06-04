import os.path
import numpy
import torch
import torch.distributed as dist
import cv2
import time
import torchvision
from typing import Optional

from tqdm import tqdm
from collections import OrderedDict

from utils import visualizer
from .visz_utils import DrawResultBboxesAndKeyptsOnStereoEventFrame, RenderImageWithBboxes
from ..models.utils.misc import freeze_module_grads, convert_tensor_to_numpy
from ..methods.visz_utils import RenderImageWithBboxesAndKeypts
from .log_utils import GetLogDict
from..models.utils.misc import freeze_module_grads

import logging
logger = logging.getLogger(__name__)


def batch_to_cuda(batch_data, dtype=torch.float32):
    def _batch_to_cuda(batch_data, dtype):
        if isinstance(batch_data, dict):
            for key in batch_data.keys():
                batch_data[key] = _batch_to_cuda(batch_data[key], dtype=dtype)
        elif isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.to(dtype).cuda()
        elif isinstance(batch_data, numpy.ndarray):
            batch_data = torch.from_numpy(batch_data).to(dtype).cuda()
        elif isinstance(batch_data, list):
            for ii, oneElement in enumerate(batch_data):
                batch_data[ii] = _batch_to_cuda(oneElement, dtype)
        elif batch_data is None:
            batch_data = batch_data
        else:
            import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()
            raise NotImplementedError

        return batch_data

    if "imagedata" in batch_data.keys() and batch_data["imagedata"] is not None:
        batch_data["imagedata"] = batch_data["imagedata"].to(dtype).cuda()

    if "objdet" in batch_data:
        batch_data["objdet"] = _batch_to_cuda(batch_data["objdet"], dtype)

    if "gt_labels" in batch_data:
        batch_data["gt_labels"]['objdet'] = _batch_to_cuda(batch_data["gt_labels"]['objdet'], dtype)
    return batch_data


def freeze_static_components(models: dict):
    """
    freeze its gradients, if no need to train it.
    """
    for key, model in models.items():
        if model.module.is_freeze:
            logger.info("---- freeze params for {}".format(key))
            freeze_module_grads(models[key])


def _forward_one_batch(
    model: torch.nn.Module,
    model_inputs: dict,
    labels: Optional[torch.Tensor],
    lossDictAll: Optional[dict],
    necessary_info: dict, 
    scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler] = None
):
    starttime = time.time()
    artifacts = None
    if scaler is not None:
        with torch.autocast(device_type="cuda", cache_enabled=True):
            preds, losses, artifacts = model(**model_inputs, labels=labels, **necessary_info)
    else:
        preds, losses, artifacts = model(**model_inputs, labels=labels, **necessary_info)
    if losses is not None:
        lossDictAll.update(losses)  # Note: losses[0] is a dict of all losses
    logger.debug("-> forward ({}) time cost: {}".format(type(model).__name__, time.time() - starttime))
    return preds, lossDictAll, artifacts


def _backward_and_optimize(
    models: dict,
    lossDictCurrentStep: dict,
    optimizer: dict,  # dict containing sub optimzers
    lossRecords: dict,
    batchSize: int,
    clip_max_norm: Optional[float] = None,  # param used for amp
    scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler] = None,
):
    starttime = time.time()
    loss = 0
    for key, value in lossDictCurrentStep.items():
        loss += value
        if key in lossRecords:
            lossRecords[key].update(lossDictCurrentStep[key].item(), batchSize)
    lossRecords["BestIndex"].update(loss.item(), batchSize)
    lossRecords["Loss"].update(loss.item(), batchSize)

    if scaler is not None:
        scaler.scale(loss).backward()
        if clip_max_norm > 0:
            for key, suboptimizer in optimizer.items():
                scaler.unscale_(suboptimizer)
            for model in models.values():
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        for key, suboptimizer in optimizer.items():
            if not models[key].module.is_freeze:
                scaler.step(suboptimizer)
        scaler.update()
    else:
        loss.backward()  # Note: PyTorch’s autograd engine ensures that gradients are only computed for parameters that contribute to a given loss term.
        for key, suboptimizer in optimizer.items():
            if not models[key].module.is_freeze:
                suboptimizer.step()
    logger.debug("-> backward_and_optimize time cost: {}".format(time.time() - starttime))
    return


def train(
    models,  # dict of models used in stereoeventobjectdetection task
    data_loader,
    optimizer,
    tensorBoardLogger,
    scaler=None,  # Note: GradScaler used for auto mixed precision
    ema=None,
    clip_max_norm=None,
    is_distributed=False,
    world_size=1,
    epoch=None
):
    """
    Args:
        ...
        ema: Exponential Moving Average. Smoothing between epoches.
    """
    for model in models.values():
        model.train()

    log_dict = GetLogDict()
    lossDictAll = {}

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader)):
        batch_data = next(data_iter)

        if "bboxes" not in batch_data["gt_labels"]["objdet"][0]:
            print("Error: the batch data do not contain GT for bboxes.")
            continue
        batch_data = batch_to_cuda(batch_data)

        # classes labels in objdet need to be int
        for indexObj in range(len(batch_data['gt_labels']['objdet'])):
            batch_data['gt_labels']['objdet'][indexObj]['labels'] = batch_data['gt_labels']['objdet'][indexObj]['labels'].to(torch.long)
            batch_data['objdet'][indexObj]['labels'] = batch_data['objdet'][indexObj]['labels'].to(torch.long)

        for key, suboptimizer in optimizer.items():
            if not models[key].module.is_freeze:
                suboptimizer.zero_grad()

        imageHeight, imageWidth = batch_data["imagedata"].shape[-2:]

        # ---------- detr net ----------
        gt_labels_forrtdetr = []
        for onegt in batch_data["gt_labels"]["objdet"]: 
            onegt = {
                "bboxes": torchvision.ops.box_convert(onegt["bboxes"][..., :4].clone(), in_fmt="xyxy", out_fmt="cxcywh"),  # Note: only left bbox is needed for detr
                "labels": onegt["labels"].clone()
            }
            onegt["bboxes"][:, [0, 2]] /= imageWidth
            onegt["bboxes"][:, [1, 3]] /= imageHeight
            gt_labels_forrtdetr.append(onegt)
        global_step = epoch * len(data_loader) + indexBatch
        epoch_info = dict(epoch=epoch, step=indexBatch, global_step=global_step)
        detections, lossDictAll, artifacts = _forward_one_batch(
            models["rtdetr"],
            {
                "x": batch_data["imagedata"],
                "is_test": False  # Note: the output will be different in test for exporting onnx model.
            },
            gt_labels_forrtdetr,
            lossDictAll,
            epoch_info,
            scaler if not models["rtdetr"].module.is_freeze else None
        )
        selected_detections, corresponding_gt_labels, indices = artifacts

        # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
        if tensorBoardLogger is not None:
            bboxes = selected_detections[0]["bboxes"]
            bboxes = torchvision.ops.box_convert(bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
            bboxes[:, [0, 2]] *= imageWidth
            bboxes[:, [1, 3]] *= imageHeight
            image_visz = RenderImageWithBboxes(
                batch_data["imagedata"][0, 0].detach().squeeze(1).cpu().numpy(),
                {
                    "bboxes": bboxes,
                    "classes": selected_detections[0]["classes"],
                }
            )
            tensorBoardLogger.add_image("(train) image with bboxes", image_visz[0])
            bboxes = gt_labels_forrtdetr[0]["bboxes"]
            bboxes = torchvision.ops.box_convert(bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
            bboxes[:, [0, 2]] *= imageWidth
            bboxes[:, [1, 3]] *= imageHeight
            image_visz = RenderImageWithBboxes(
                batch_data["imagedata"][0, 0].detach().squeeze(1).cpu().numpy(),
                {
                    "bboxes": bboxes,
                    "classes": gt_labels_forrtdetr[0]["labels"],
                }
            )
            tensorBoardLogger.add_image("(train) image with GT bboxes", image_visz[0])
            # image_visz = RenderImageWithBboxes(
            #     batch_data["imagedata"][0, 1].detach().squeeze(1).cpu().numpy(),
            #     {
            #         "bboxes": bboxes,
            #         "classes": gt_labels_forrtdetr[0]["labels"],
            #     }
            # )
            # tensorBoardLogger.add_image("(train) normals with GT bboxes", image_visz[0])

        # backward and optimize
        batchSize = batch_data["imagedata"].shape[0]
        _backward_and_optimize(
            models,
            lossDictAll,
            optimizer,  # dict containing sub optimzers
            log_dict,
            batchSize,
            clip_max_norm,  # param used for amp
            scaler if not models["rtdetr"].module.is_freeze else None  # Note: only training detr we need this.
        )

        if ema is not None:
            # exponential moving average
            for key, model in models.items():
                if not model.module.is_freeze:
                    ema[key].update(model)

        if tensorBoardLogger is not None:
            pbar.update(1)
        torch.cuda.synchronize()

    if tensorBoardLogger is not None:
        pbar.close()
    return log_dict


@torch.no_grad()
def valid(
    models,
    data_loader,
    is_distributed=False,
    world_size=1,
    tensorBoardLogger=None,
    epoch=None
):
    """
    Args:
        ...
        ema: Exponential Moving Average. Smoothing between epoches.
    """
    for model in models.values():
        model.eval()

    log_dict = GetLogDict()
    lossDictAll = {}

    if tensorBoardLogger is not None:
        pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader)):
        batch_data = next(data_iter)

        if "bboxes" not in batch_data["gt_labels"]["objdet"][0]:
            print("Error: the batch data do not contain GT for bboxes.")
            continue

        batch_data = batch_to_cuda(batch_data)

        # classes labels in objdet need to be int
        for indexObj in range(len(batch_data['gt_labels']['objdet'])):
            batch_data['gt_labels']['objdet'][indexObj]['labels'] = batch_data['gt_labels']['objdet'][indexObj]['labels'].to(torch.long)
            batch_data['objdet'][indexObj]['labels'] = batch_data['objdet'][indexObj]['labels'].to(torch.long)

        imageHeight, imageWidth = batch_data['imagedata'].shape[-2:]

        # ---------- detr net ----------
        gt_labels_forrtdetr = []
        for onegt in batch_data["gt_labels"]["objdet"]: 
            onegt = {
                "bboxes": torchvision.ops.box_convert(onegt["bboxes"][..., :4].clone(), in_fmt="xyxy", out_fmt="cxcywh"),  # Note: only left bbox is needed for detr
                "labels": onegt["labels"].clone()
            }
            onegt["bboxes"][:, [0, 2]] /= imageWidth
            onegt["bboxes"][:, [1, 3]] /= imageHeight
            gt_labels_forrtdetr.append(onegt)
        global_step = epoch * len(data_loader) + indexBatch
        epoch_info = dict(epoch=epoch, step=indexBatch, global_step=global_step)
        detections, lossDictAll, artifacts = _forward_one_batch(
            models["rtdetr"],
            {
                "x": batch_data["imagedata"],
                "is_test": False
            },
            gt_labels_forrtdetr,
            lossDictAll,
            epoch_info
        )
        selected_detections, corresponding_gt_labels, indices = artifacts

        # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
        if tensorBoardLogger is not None:
            bboxes = selected_detections[0]["bboxes"]
            bboxes = torchvision.ops.box_convert(bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
            bboxes[:, [0, 2]] *= imageWidth
            bboxes[:, [1, 3]] *= imageHeight
            image_visz = RenderImageWithBboxes(
                batch_data["imagedata"][0, 0].detach().squeeze(1).cpu().numpy(),
                {
                    "bboxes": bboxes,
                    "classes": selected_detections[0]["classes"],
                }
            )
            tensorBoardLogger.add_image("(valid) image with bboxes", image_visz[0])
            bboxes = gt_labels_forrtdetr[0]["bboxes"]
            bboxes = torchvision.ops.box_convert(bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
            bboxes[:, [0, 2]] *= imageWidth
            bboxes[:, [1, 3]] *= imageHeight
            image_visz = RenderImageWithBboxes(
                batch_data["imagedata"][0, 0].detach().squeeze(1).cpu().numpy(),
                {
                    "bboxes": bboxes,
                    "classes": gt_labels_forrtdetr[0]["labels"],
                }
            )
            tensorBoardLogger.add_image("(valid) image with GT bboxes", image_visz[0])
            # image_visz = RenderImageWithBboxes(
            #     batch_data["imagedata"][0, 1].detach().squeeze(1).cpu().numpy(),
            #     {
            #         "bboxes": bboxes,
            #         "classes": gt_labels_forrtdetr[0]["labels"],
            #     }
            # )
            # tensorBoardLogger.add_image("(valid) normals with GT bboxes", image_visz[0])


        batchSize = batch_data["imagedata"].shape[0]
        loss = 0
        for key, value in lossDictAll.items():
            loss += value
            if key in log_dict:
                log_dict[key].update(lossDictAll[key].item(), batchSize)
        log_dict["BestIndex"].update(loss.item(), batchSize)
        log_dict["Loss"].update(loss.item(), batchSize)

        if tensorBoardLogger is not None:
            pbar.update(1)

    torch.cuda.synchronize()

    if tensorBoardLogger is not None:
        pbar.close()

    return log_dict


@torch.no_grad()
def test(
    models,
    data_loader,
    dataset_name,
    save_root,
    is_save_onnx = False
):
    for model in models.values():
        model.eval()

    if is_save_onnx:
        logger.info(
            '''
            # how to do inference with the exported onnx model:
            # providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            import onnxruntime
            providers = ["CPUExecutionProvider"]
            ort_session = onnxruntime.InferenceSession(os.path.join(save_root, "rtdetr.onnx"), providers=providers)
            ort_inputs = {"x": batch_data["imagedata"].cpu().numpy()}
            ort_outs = ort_session.run(["pred_logits", "pred_boxes"], ort_inputs)
            '''
        )

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader.dataset)):            
        batch_data = batch_to_cuda(next(data_iter))
        starttime = time.time()
        detections = _forward_one_batch(
            models["rtdetr"],
            {
                "x": batch_data["imagedata"]
            },
            None,
            None,
            {}
        )[0]
        print("one infer time: {}".format(time.time() - starttime))
        if is_save_onnx:
            torch.onnx.export(
                models['rtdetr'],
                (
                    batch_data["imagedata"]
                ),
                os.path.join(save_root, "rtdetr.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["x"],
                output_names=["pred_logits", "pred_boxes"]
            )
        
        scoreThreshold = 0.5
        SaveTestResults(
            {"pred_logits": detections[0], "pred_boxes": detections[1]},
            scoreThreshold,
            batch_data["imagedata"],
            indexBatch,
            batch_data["data_index"].item(),
            dataset_name,
            save_root,
            batch_data["image_metadata"],
            model.__class__.__name__
        )
        pbar.update(1)
    pbar.close()
    return


def SaveTestResults(
    detections,
    scoreThreshold,
    imagedata,
    indexBatch,
    data_index,
    dataset_name,
    save_root,
    image_metadata,
    model_name
):
    """
    Save the test results to the disk.
    """
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_path = os.path.join(save_root, dataset_name, "test", "preds_visz")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image = imagedata[0, 0].detach().cpu().numpy() * 255
    image = image.astype(numpy.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for indexInBatch in range(detections["pred_boxes"].shape[0]):
        bboxes = detections["pred_boxes"][indexInBatch]
        bboxes = torchvision.ops.box_convert(bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
        bboxes[:, [0, 2]] *= image_metadata["w"]
        bboxes[:, [1, 3]] *= image_metadata["h"]
        scores, labels = detections["pred_logits"][indexInBatch].softmax(dim=-1).max(dim=-1)
        for bbox, score, label in zip(bboxes, scores, labels):
            if score < scoreThreshold:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            label_text = f"{label.item()}:{score.item():.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imwrite(os.path.join(save_path, f"{data_index:06d}.png"), image)
