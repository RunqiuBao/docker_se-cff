import partial
from typing import Union
import torch.nn as nn


def freeze_module_grads(module: Union(nn.Module, nn.ModuleList)):
    """Freeze the gradients of a module or modules in a module list."""
    if isinstance(module, nn.ModuleList):
        for m in module:
            freeze_module_grads(m)
    if isinstance(module, nn.Module):
        for param in module.parameters():
            param.requires_grad = False
    return


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
