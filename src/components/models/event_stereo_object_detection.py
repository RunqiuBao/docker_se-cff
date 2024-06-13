import torch.nn as nn
import torch
from thop import profile

from .concentration import ConcentrationNet
from .stereo_matching import StereoMatchingNetwork
from .objectdetection import Cylinder5DDetectionHead
from .feature_extractor import FeatureExtractor

from ...utils import losses as losses
from ...utils import metrics as metrics


class EventStereoObjectDetectionNet(nn.Module):
    _concentration_net = None
    _feature_extraction_net = None
    _object_detection_head = None
    _disp_head = None
    _objdet_loss = None
    _disp_loss = None

    def __init__(
        self,
        concentration_net_cfg=None,
        feature_extraction_net_cfg=None,
        disp_head_cfg=None,
        object_detection_head_cfg=None,
        loss_cfg=None,  # Cylinder5DDetectionLoss, SparseDispLoss
        metric_cfg=None,
        logger=None,
    ):
        super(EventStereoObjectDetectionNet, self).__init__()
        self._concentration_net = ConcentrationNet(**concentration_net_cfg.PARAMS)
        self._feature_extraction_net = FeatureExtractor(
            **feature_extraction_net_cfg.PARAM
        )
        self._object_detection_head = Cylinder5DDetectionHead(
            **object_detection_head_cfg.PARAM
        )
        self._disp_head = StereoMatchingNetwork(
            **disp_head_cfg.PARAM, isInputFeature=True
        )

        self._objdet_loss = getattr(losses, loss_cfg.OBJDET.NAME)(
            loss_cfg.OBJDET.PARAMS
        )
        self._disp_loss = getattr(losses, loss_cfg.DISP.NAME)(loss_cfg.DISP.PARAMS)

    def forward(self, left_event, right_event, gt_labels=None):
        """
        Args:
            left/right_event: (b c s t h w) tensor
            gt_labels: ?
        """
        event_stack = {"l": left_event, "r": right_event}

        concentrated_event_stack = {}
        for loc in ["l", "r"]:
            concentrated_event_stack[loc] = self._concentration_net(event_stack[loc])

        left_feature = self._feature_extraction_net(
            concentrated_event_stack["l"]
        )  # Note: multi-scale
        right_feature = self._feature_extraction_net(
            concentrated_event_stack["r"]
        )  # Note: multi-scale

        pred_disparity_pyramid = self._disp_head(left_feature, right_feature)
        object_detection_prediction = self._object_detection_head(
            left_feature, right_feature, pred_disparity_pyramid
        )

        loss_objdet = None
        metric_objdet = None
        if gt_labels is not None:
            loss_objdet = self._cal_loss(
                object_detection_prediction, pred_disparity_pyramid, gt_labels
            )
            # metric_objdet = self._cal_metric(object_detection_prediction, gt_labels)

        return object_detection_prediction, pred_disparity_pyramid, loss_objdet

    def _cal_loss(self, object_detection_prediction, gt_labels):
        pass

    def _cal_loss_disp(self, pred_disparity_pyramid, gt_disparity):
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]

        loss = 0.0
        mask = gt_disparity > 0
        for idx in range(len(pred_disparity_pyramid)):
            pred_disp = pred_disparity_pyramid[idx]
            weight = pyramid_weight[idx]

            if pred_disp.size(-1) != gt_disparity.size(-1):
                pred_disp = pred_disp.unsqueeze(1)
                pred_disp = F.interpolate(
                    pred_disp,
                    size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                    mode="bilinear",
                    align_corners=False,
                ) * (gt_disparity.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)

            cur_loss = self.criterion(
                pred_disp[mask], gt_disparity[mask]
            )  # Note: gt_disparity was not normalized
            loss += weight * cur_loss

        return loss

    def _cal_metric(object_detection_prediction, gt_labels):
        pass

    @staticmethod
    def ComputeProfile(model, inputShape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        left_event = torch.randn(*inputShape).to(device)
        right_event = torch.randn(*inputShape).to(device)
        model = model.to(device)
        flops, numParams = profile(
            model, input=(left_event, right_event), verbose=False
        )
        return flops, numParams
