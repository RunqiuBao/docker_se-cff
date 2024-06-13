import torch.nn


class DisparityLoss(torch.nn.Module):
    _smoothL1 = None

    def __init__(self):
        super(DisparityLoss, self).__init__()
        self._smoothL1 = nn.SmoothL1Loss(reduction="none")
