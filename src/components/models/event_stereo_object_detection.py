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
            metric_cfg=None):
        super(EventStereoObjectDetectionNet, self).__init__()
        self._concentration_net = ConcentrationNet(**concentration_net_cfg.PARAMS)
        self._feature_extraction_net = FeatureExtractor(**feature_extraction_net_cfg.PARAM)
        self._object_detection_head = Cylinder5DDetectionHead(**object_detection_head_cfg.PARAM)
        self._disp_head = StereoMatchingNetwork(**disp_head_cfg.PARAM)

        self._objdet_loss = getattr(losses, loss_cfg.OBJDET.NAME)(loss_cfg.OBJDET.PARAMS)
        self._disp_loss = getattr(losses, loss_cfg.DISP.NAME)(loss_cfg.DISP.PARAMS)

    def forward(self, left_event, right_event, gt_labels=None):
        """
        Args:
            left/right_event: (b c s t h w) tensor
            gt_labels: ?
        """
        event_stack = {
            'l': left_event,
            'r': right_event
        }

        concentrated_event_stack = {}
        for loc in ['l', 'r']:
            concentrated_event_stack[loc] = self._concentration_net(event_stack[loc])

        left_feature = self._feature_extraction_net(left_event)  # Note: multi-scale
        right_feature = self._feature_extraction_net(right_event)  # Note: multi-scale

        pred_disparity_pyramid = self._disp_head(left_feature, right_feature)
        object_detection_prediction = self._object_detection_head(left_feature, right_feature, pred_disparity_pyramid)

        loss_objdet = None
        metric_objdet = None
        if gt_labels is not None:
            loss_objdet = self._cal_loss(object_detection_prediction, pred_disparity_pyramid, gt_labels)
            # metric_objdet = self._cal_metric(object_detection_prediction, gt_labels)

        return object_detection_prediction, pred_disparity_pyramid, loss_objdet

    def _cal_loss(self, object_detection_prediction, gt_labels):
        pass

    def _cal_metric(object_detection_prediction, gt_labels):
        pass

    @staticmethod
    def ComputeProfile(model,  inputShape):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        left_event = torch.randn(*inputShape).to(device)
        right_event = torch.randn(*inputShape).to(device)
        model = model.to(device)
        flops, numParams = profile(model, input=(left_event, right_event), verbose=False)
        return flops, numParams

