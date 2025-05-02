from collections import OrderedDict
from utils.metrics import AverageMeter


def GetLogDict():
    return OrderedDict(
        [
            ("BestIndex", AverageMeter(string_format="%6.3lf")),
            ("Loss", AverageMeter(string_format="%6.3lf")),
            ("loss_rtdetr", AverageMeter(string_format="%6.3lf"))
        ]
    )