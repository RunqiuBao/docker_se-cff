import numpy
import math
import torch
import torch.nn as nn
import copy
from typing import Sequence, Tuple, List, Dict, Union, Optional
from torch import Tensor
from thop import profile

from torch.nn.modules.upsampling import Upsample
from .yolo_pose_blocks import Conv, C3k2, Concat, SPPF, C2PSA, DWConv, DFL
from .yolo_pose_utils import make_anchors, dist2bbox, initialize_weights
from .losses import v8PoseLoss

import logging
logger = logging.getLogger(__name__)


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""        
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            
            return y


class YoloPose(nn.Module):
    """
    bbox and keypoint detection by yolo model.
    """
    _criterion = None  # loss func

    def __init__(
        self,
        network_cfg: dict,
        loss_cfg: dict,
        is_freeze: bool,
        **kwargs
    ):
        super(YoloPose, self).__init__()
        self._config = network_cfg
        self._config["is_freeze"] = is_freeze
        self._config["loss_cfg"] = loss_cfg
        
        layers = []
        modules_to_use = [Conv, Conv, C3k2, Conv, C3k2, Conv, C3k2, Conv, C3k2, SPPF, C2PSA, Upsample, Concat, C3k2, Upsample, Concat, C3k2, Conv, Concat, C3k2, Conv, Concat, C3k2, Pose]
        args_to_use = [
            [self._config["in_channels"], 16, 3, 2],
            [16, 32, 3, 2],
            [32, 64, 1, False, 0.25],
            [64, 64, 3, 2],
            [64, 128, 1, False, 0.25],
            [128, 128, 3, 2],
            [128, 128, 1, True],
            [128, 256, 3, 2],
            [256, 256, 1, True],
            [256, 256, 5],
            [256, 256, 1],
            [None, 2, 'nearest'],
            [1],
            [384, 128, 1, False],
            [None, 2, 'nearest'],
            [1],
            [256, 128, 1, False],
            [128, 64, 3, 2],
            [1],
            [192, 128, 1, False],
            [128, 128, 3, 2],
            [1],
            [384, 128, 1, True],
            [1, [2, 3], [128, 128, 128]]
        ]
        froms = [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            [-1, 6],
            -1,
            -1,
            [-1, 4],
            -1,
            -1,
            [-1, 13],
            -1,
            -1,
            [-1, 10],
            -1,
            [16, 19, 22]
        ]

        for indexModule, (module_name, args, f) in enumerate(zip(modules_to_use, args_to_use, froms)):
            m_ = module_name(*args)
            t = str(module_name)[8:-2].replace("__main__.", "")  # module type
            m_.np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type = indexModule, f, t  # attach index, 'from' index, type
            layers.append(m_)

        self._model = torch.nn.Sequential(*layers)
        self._save = [4, 6, 10, 13, 16, 19, 22]  # used in predict.
        self._inplace = True
        self._end2end = getattr(self._model[-1], "end2end", False)

        # Build strides
        m = self._model[-1]  # Detect()
        s = 256  # 2x min stride
        m.inplace = self._inplace
        def _forward(x):
            """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
            if self._end2end:
                return self.forward(x)["one2many"]
            return self.forward(x)[0][0]
        
        m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, self._config["in_channels"], s, s))])  # forward
        self.stride = m.stride
        m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""        
        self._criterion = v8PoseLoss(self)
        return
    
    @property
    def config(self):
        return self._config

    @property
    def is_loss_ready(self):
        return self._criterion is not None

    @property
    def is_freeze(self):
        return self._config["is_freeze"]
    
    @property
    def loss_cfg(self):
        return self._config["loss_cfg"]
    
    @property
    def model(self):
        return self._model

    @property
    def input_shape(self):
        return
    
    @staticmethod
    def ComputeCostProfile(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        left_event_voxel = torch.randn(4, 1, 480, 672).to(device)
        right_event_voxel = torch.randn(4, 1, 480, 672).to(device)
        model = model.to(device)
        flops, numParams = profile(model, inputs=(left_event_voxel, right_event_voxel), verbose=False)
        return flops, numParams

    def predict(
        self,
        x: Tensor,
        isRightFeatures: Optional[bool] = None
    ):
        y = []  # outputs
        if isRightFeatures:
            indicesToRetrieveFeatures = [16, 19, 22]
            intermediateFeatures = []

        for indexModule, m in enumerate(self._model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                
            x = m(x)  # run
            if isRightFeatures:
                if indexModule in indicesToRetrieveFeatures:
                    intermediateFeatures.append(x)
                if indexModule == indicesToRetrieveFeatures[-1]:
                    x = intermediateFeatures
                    break
            y.append(x if m.i in self._save else None)  # save for using in following layers

        return x

    def forward(
        self,
        left_event_voxel: Tensor,
        right_event_voxel: Optional[Tensor] = None,
        labels = None,
        **kwargs
    ):
        """
        forward to train a model. Return GT aligned predictions and losses.

        Args:
            left_event_voxel: left event input. shape [B, 10, h, w].
            right_event_voxel: right event input. shape [B, 10, h, w].
            ...
            gt_labels: list of dict for one batch. each dict contains 'bboxes' and 'labels' keys.
        Returns:
            preds: GT aligned predictions.
            loss_dict_final: losses.
        """
        preds = self.predict(left_event_voxel)

        right_features = None
        if right_event_voxel is not None:
            isTraining = self.training
            if isTraining:
                self.eval()
            with torch.no_grad():
                right_features = self.predict(right_event_voxel, isRightFeatures=True)
            if isTraining:
                self.train()

        losses = None
        artifacts = [right_features, None, None, None, None, None, None, None]
        if labels is not None:
            losses_and_artifacts = self.compute_loss(preds, labels)
            artifacts[1] = losses_and_artifacts[1]
            artifacts[2] = losses_and_artifacts[2]
            artifacts[3] = losses_and_artifacts[3]
            artifacts[4] = losses_and_artifacts[4]
            artifacts[5] = losses_and_artifacts[5]
            artifacts[6] = losses_and_artifacts[6]
            artifacts[7] = losses_and_artifacts[7]
            losses = losses_and_artifacts[0]
            if self.is_freeze:
                for key, loss_value in losses.items():
                    losses[key] *= 0
        return preds, losses, artifacts
    
    def compute_loss(
        self,
        preds: Tuple[List, Tensor],
        labels: Dict
    ):
        return self._criterion(preds, labels)



