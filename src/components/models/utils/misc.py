from functools import partial
from typing import Union
import time
import torch
import torch.nn as nn
from torch import Tensor
import copy
import cv2

def freeze_module_grads(module: Union[nn.Module, nn.ModuleList]):
    """Freeze the gradients of a module or modules in a module list."""
    if isinstance(module, nn.ModuleList):
        for m in module:
            freeze_module_grads(m)
    if isinstance(module, nn.Module):
        for param in module.parameters():
            param.requires_grad = False
    return

def unfreeze_module_grads(module: Union[nn.Module, nn.ModuleList]):
    """unfreeze the gradients of a module or modules in a module list."""
    if isinstance(module, nn.ModuleList):
        for m in module:
            freeze_module_grads(m)
    if isinstance(module, nn.Module):
        for param in module.parameters():
            param.requires_grad = True
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


def SelectByIndices(original: Tensor, indices: Tensor):
    """
    Args:
        original: shape (B, N) or (B, N, C)
        indices: shape (B, K)
    """
    if original.dim() == indices.dim() + 1:
        indices_expand = indices.unsqueeze(-1).expand(-1, -1, original.shape[-1])
        selected = torch.gather(original, 1, indices_expand)
        return selected
    elif original.dim() == indices.dim():
        selected = torch.gather(original, 1, indices)
        return selected
    else:
        raise ValueError("original size ({}) not matching indices size ({})".format(original.shape, indices.shape))


def ComputeSoftArgMax1d(input, beta=100):
    """
    assume input to be 3d tensor. spatial distribution is on the last dimension.
    """
    n = input.shape[-1]
    input = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n).to(input.device)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def DetachCopyNested(tensorList):
    """
    detach and copy a nested list of tensor. prepare input for next network.
    """
    newList = []
    for oneEntry in tensorList:
        if isinstance(oneEntry, Tensor):
            newList.append(copy.deepcopy(oneEntry.detach()))
        else:
            newList.append(DetachCopyNested(oneEntry))
    return newList


def generate_gaussian(size, center, std=0.8):
    """
    Generate a 2D Gaussian probability distribution centered at a given point.
    
    Parameters:
    - size (int): The size of the output tensor (size x size).
    - center (Tensor): The coordinates of the center point (x, y). In shape of [B, 2]
    - std (float): The standard deviation of the Gaussian distribution. Default is 1.0.
    
    Returns:
    - gaussian (torch.Tensor): The generated Gaussian distribution of shape (B, size, size).
    """
    x = torch.arange(0, size, dtype=torch.float32, device=center.device)
    y = torch.arange(0, size, dtype=torch.float32, device=center.device)
    yy, xx = torch.meshgrid(x, y)
    
    batch_size = center.shape[0]
    xx = xx.view(1, size, size).expand(batch_size, -1, -1)
    yy = yy.view(1, size, size).expand(batch_size, -1, -1)

    # Calculate the Gaussian distribution
    gaussian = torch.exp(-((xx - center[:, 0].view(-1, 1, 1).expand(-1, size, size))**2 + (yy - center[:, 1].view(-1, 1, 1).expand(-1, size, size))**2) / (2 * std**2))

    # Normalize to make it a valid probability distribution
    gaussian /= gaussian.view(batch_size, -1).sum(dim=1).view(-1, 1, 1).expand(-1, size, size)        

    return gaussian
