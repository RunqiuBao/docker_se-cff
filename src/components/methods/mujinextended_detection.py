import os.path
import numpy
import torch
import torch.distributed as dist
import torch.nn.functional as F
import cv2
import time
import torchvision
from typing import Optional
import random

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
        elif isinstance(batch_data, (int, str, float, None)):
            batch_data = batch_data
        else:
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


def compute_multi_scale_scales(resolution: int, expanded_scales: bool = False, patch_size: int = 16, num_windows: int = 4) -> list[int]:
    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]  # ensure minimum image size
    return proposed_scales


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

        if "boxes" not in batch_data["gt_labels"]["objdet"][0]:
            print("Error: the batch data do not contain GT for bboxes.")
            continue
        batch_data = batch_to_cuda(batch_data)

        for key, suboptimizer in optimizer.items():
            if not models[key].module.is_freeze:
                suboptimizer.zero_grad()

        # random multi_scale implemented in rfdetr
        scales = compute_multi_scale_scales(
            data_loader.dataset.rfdetr_resolution,
            True,
            data_loader.dataset.rfdetr_patch_size,
            data_loader.dataset.rfdetr_num_windows
        )
        it = epoch * len(data_loader) + indexBatch
        random.seed(it)
        scale = random.choice(scales)
        with torch.no_grad():
            batch_data["imagedata"] = F.interpolate(batch_data["imagedata"], size=scale, mode='bilinear', align_corners=False)
            for indexInBatch in range(batch_data["gt_labels"]["objdet"].__len__()):
                oneLabel = batch_data["gt_labels"]["objdet"][indexInBatch]
                if "masks" in oneLabel:
                    batch_data["gt_labels"]["objdet"][indexInBatch]["masks"] = F.interpolate(oneLabel["masks"].unsqueeze(1).float(), size=scale, mode='nearest').squeeze(1).bool()

        imageHeight, imageWidth = batch_data["imagedata"].shape[-2:]
        # ---------- detr net ----------
        global_step = epoch * len(data_loader) + indexBatch
        epoch_info = dict(epoch=epoch, step=indexBatch, global_step=global_step)
        detections, lossDictAll, artifacts = _forward_one_batch(
            models["rfdetr"],
            {
                "x": batch_data["imagedata"],
            },
            batch_data['gt_labels']['objdet'],
            lossDictAll,
            epoch_info,
            scaler if not models["rfdetr"].module.is_freeze else None
        )
        selected_detections = artifacts[0]

        # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
        if tensorBoardLogger is not None:
            bboxes = selected_detections["bboxes"]
            bboxes = torchvision.ops.box_convert(bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
            bboxes[:, [0, 2]] *= imageWidth
            bboxes[:, [1, 3]] *= imageHeight
            image_visz = RenderImageWithBboxes(
                batch_data["imagedata"][0, 0].detach().squeeze(1).cpu().numpy(),
                {
                    "bboxes": bboxes,
                    "classes": selected_detections["classes"],
                }
            )
            tensorBoardLogger.add_image("(train) image with bboxes", image_visz[0])
            bboxes = batch_data['gt_labels']['objdet'][0]["boxes"]
            bboxes = torchvision.ops.box_convert(bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
            bboxes[:, [0, 2]] *= imageWidth
            bboxes[:, [1, 3]] *= imageHeight
            image_visz = RenderImageWithBboxes(
                batch_data["imagedata"][0, 0].detach().squeeze(1).cpu().numpy(),
                {
                    "bboxes": bboxes,
                    "classes": batch_data['gt_labels']['objdet'][0]["labels"],
                }
            )
            tensorBoardLogger.add_image("(train) image with GT bboxes", image_visz[0])
            if "target_masks" in selected_detections and "predicted_masks" in selected_detections:
                targetMaskSample = selected_detections["target_masks"][0, 0].sigmoid()
                predictedMaskSample = selected_detections["predicted_masks"][0].unsqueeze(0)
                predictedMaskSample = F.interpolate(predictedMaskSample, size=targetMaskSample.shape[-2], mode='bilinear', align_corners=True)
                predictedMaskSample = predictedMaskSample.sigmoid()[0, 0]
                predictedMaskSample = 1 - predictedMaskSample  # invert for better visualization
                image_visz = torch.cat([targetMaskSample, predictedMaskSample], dim=1) * 255.0
                image_visz = image_visz.detach().cpu().to(dtype=torch.uint8)
                tensorBoardLogger.add_image("(train) target and predicted masks sample", image_visz)

            # image_visz = RenderImageWithBboxes(
            #     batch_data["imagedata"][0, 1].detach().squeeze(1).cpu().numpy(),
            #     {
            #         "bboxes": bboxes,
            #         "classes": gt_labels_forrfdetr[0]["labels"],
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
            scaler if not models["rfdetr"].module.is_freeze else None  # Note: only training detr we need this.
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

        if "boxes" not in batch_data["gt_labels"]["objdet"][0]:
            print("Error: the batch data do not contain GT for bboxes.")
            continue

        batch_data = batch_to_cuda(batch_data)

        imageHeight, imageWidth = batch_data['imagedata'].shape[-2:]

        # ---------- detr net ----------
        global_step = epoch * len(data_loader) + indexBatch
        epoch_info = dict(epoch=epoch, step=indexBatch, global_step=global_step)
        detections, lossDictAll, artifacts = _forward_one_batch(
            models["rfdetr"],
            {
                "x": batch_data["imagedata"],
            },
            batch_data["gt_labels"]["objdet"],
            lossDictAll,
            epoch_info
        )
        selected_detections = artifacts[0]

        # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
        if tensorBoardLogger is not None:
            bboxes = selected_detections["bboxes"]
            bboxes = torchvision.ops.box_convert(bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
            bboxes[:, [0, 2]] *= imageWidth
            bboxes[:, [1, 3]] *= imageHeight
            image_visz = RenderImageWithBboxes(
                batch_data["imagedata"][0, 0].detach().squeeze(1).cpu().numpy(),
                {
                    "bboxes": bboxes,
                    "classes": selected_detections["classes"],
                }
            )
            tensorBoardLogger.add_image("(valid) image with bboxes", image_visz[0])
            bboxes = batch_data["gt_labels"]["objdet"][0]["boxes"]
            bboxes = torchvision.ops.box_convert(bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
            bboxes[:, [0, 2]] *= imageWidth
            bboxes[:, [1, 3]] *= imageHeight
            image_visz = RenderImageWithBboxes(
                batch_data["imagedata"][0, 0].detach().squeeze(1).cpu().numpy(),
                {
                    "bboxes": bboxes,
                    "classes": batch_data["gt_labels"]["objdet"][0]["labels"],
                }
            )
            tensorBoardLogger.add_image("(valid) image with GT bboxes", image_visz[0])
            if "target_masks" in selected_detections and "predicted_masks" in selected_detections:
                targetMaskSample = selected_detections["target_masks"][0, 0].sigmoid()
                predictedMaskSample = selected_detections["predicted_masks"][0].unsqueeze(0)
                predictedMaskSample = F.interpolate(predictedMaskSample, size=targetMaskSample.shape[-2], mode='bilinear', align_corners=True)
                predictedMaskSample = predictedMaskSample.sigmoid()[0, 0]
                predictedMaskSample = 1 - predictedMaskSample  # invert for better visualization
                image_visz = torch.cat([targetMaskSample, predictedMaskSample], dim=1) * 255.0
                image_visz = image_visz.detach().cpu().to(dtype=torch.uint8)
                tensorBoardLogger.add_image("(valid) target and predicted masks sample", image_visz)
                # image_visz = RenderImageWithBboxes(
                #     batch_data["imagedata"][0, 1].detach().squeeze(1).cpu().numpy(),
                #     {
                #         "bboxes": bboxes,
                #         "classes": gt_labels_forrfdetr[0]["labels"],
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
            ort_session = onnxruntime.InferenceSession(os.path.join(save_root, "rfdetr.onnx"), providers=providers)
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
            models["rfdetr"],
            {
                "x": batch_data["imagedata"]
            },
            None,
            None,
            {}
        )[0]
        print("one infer time: {}".format(time.time() - starttime))
        if is_save_onnx:
            models['rfdetr'].cpu()
            batch_data["imagedata"] = batch_data["imagedata"].cpu()
            output_file = export_onnx(
                output_dir=save_root,
                model=models['rfdetr'],
                input_names=["x"],
                input_tensors=batch_data["imagedata"],
                output_names=["pred_boxes", "pred_logits"],
                dynamic_axes=None,
                backbone_only=False,
                verbose=True,
                opset_version=17,
            )
            # inference once with the exported onnx model
            import onnxruntime as ort
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(output_file, providers=providers)
            input_name = session.get_inputs()[0].name
            output_names = [output.name for output in session.get_outputs()]
            starttime = time.time()
            results = session.run(output_names, {input_name: batch_data["imagedata"].cpu().numpy()})
            print("ONNX inference time: {}".format(time.time() - starttime))
            detections["pred_logits"] = torch.from_numpy(results[1])
            detections["pred_boxes"] = torch.from_numpy(results[0])
        
        scoreThreshold = 0.5
        SaveTestResults(
            {"pred_logits": detections["pred_logits"], "pred_boxes": detections["pred_boxes"]},
            scoreThreshold,
            batch_data["imagedata"],
            indexBatch,
            batch_data["data_index"][0],
            dataset_name,
            save_root,
            {
                "h": batch_data["gt_labels"]["objdet"][0]["orig_size"][0].item(),
                "w": batch_data["gt_labels"]["objdet"][0]["orig_size"][1].item(),
            },
            model.__class__.__name__,
        )
        pbar.update(1)
        if is_save_onnx:
            print("========> ONNX model saved. Finishing.")
            break
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

    image = unnormalize(imagedata.detach().cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).numpy() * 255
    image = image.astype(numpy.uint8).squeeze(0).transpose(1, 2, 0)
    image = cv2.resize(image, (int(image_metadata["w"]), int(image_metadata["h"])), interpolation=cv2.INTER_CUBIC)
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


def export_onnx(output_dir, model, input_names, input_tensors, output_names, dynamic_axes, backbone_only=False, verbose=True, opset_version=17):
    export_name = "rfdetr_backbone_model" if backbone_only else "rfdetr_detector_model"
    output_file = os.path.join(output_dir, f"{export_name}.onnx")

    # Prepare model for export
    if hasattr(model, "export"):
        model.export()
    torch.onnx.export(
        model,
        input_tensors,
        output_file,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        verbose=verbose,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes)

    print(f'\nSuccessfully exported ONNX model: {output_file}')
    return output_file


def unnormalize(tensor, mean, std):
    """
    Args:
        tensor (torch.Tensor): The normalized image (C, H, W) or (B, C, H, W)
        mean (list): The mean used for normalization (e.g., [0.485, 0.456, 0.406])
        std (list): The std used for normalization (e.g., [0.229, 0.224, 0.225])
    """
    # 1. Convert mean/std lists to tensors
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    
    # 2. Reshape to (C, 1, 1) so it broadcasts across H and W
    # This works for both 3D (C, H, W) and 4D (B, C, H, W) tensors
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    
    # 3. Reverse the operation: (x * std) + mean
    return tensor * std + mean
