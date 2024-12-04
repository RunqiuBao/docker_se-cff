from collections import OrderedDict
from utils.metrics import AverageMeter, EndPointError, NPixelError, RootMeanSquareError


def GetLogDict(is_train: bool, is_secff: bool):
    if is_secff:
        return OrderedDict(
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
        ) 
    elif is_train:
        return OrderedDict(
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
                ("loss_facet", AverageMeter(string_format="%6.3lf")),
                ("loss_pmap", AverageMeter(string_format="%6.3lf")),
                ("loss_rtdetr", AverageMeter(string_format="%6.3lf")),
                ("loss_pickable_region", AverageMeter(string_format="%6.3lf"))
            ]
        )
    else:
        return OrderedDict(
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
                ("loss_facet", AverageMeter(string_format="%6.3lf")),
                ("loss_pmap", AverageMeter(string_format="%6.3lf")),
                ("loss_rtdetr", AverageMeter(string_format="%6.3lf")),
                ("loss_pickable_region", AverageMeter(string_format="%6.3lf"))
            ]
        )