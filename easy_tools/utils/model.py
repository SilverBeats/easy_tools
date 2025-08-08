#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Any

import numpy as np
import torch.nn as nn
from torch import Tensor

__all__ = [
    "calc_model_params",
    "freeze_model",
    "unfreeze_model",
    "clone_module",
    "data_2_device",
    "convert_data_to_normal_type",
]


def calc_model_params(model: nn.Module) -> int:
    """calculate model parameters"""
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    return total_params


def freeze_model(model: nn.Module, skip_param_names=None):
    """
    Args:
        model: model to freeze
        skip_param_names: parameter names to skip.
                In fact, if you want to skip a block, just give the block name,
                all the block parameters will be skipped.
    """
    if skip_param_names is None:
        skip_param_names = []
    for name, param in model.named_parameters():
        if any(s_p in name for s_p in skip_param_names):
            continue
        param.requires_grad = False


def unfreeze_model(model: nn.Module, skip_param_names=None):
    """
    Args:
        model: model to unfreeze
        skip_param_names: parameter names to skip.
                In fact, if you want to skip a block, just give the block name,
                all the block parameters will be skipped.
    """
    if skip_param_names is None:
        skip_param_names = []
    for name, param in model.named_parameters():
        if any(s_p in name for s_p in skip_param_names):
            continue
        param.requires_grad = True


def clone_module(module, n: int):
    """clone the module"""
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def data_2_device(data, device):
    if isinstance(data, dict):
        return {k: data_2_device(v, device) for k, v in data.items()}
    elif isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [data_2_device(item, device) for item in data]
    else:
        return data


def convert_data_to_normal_type(data: Any):
    if isinstance(data, (list, tuple)):
        return [convert_data_to_normal_type(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_data_to_normal_type(v) for k, v in data.items()}
    elif isinstance(data, (np.ndarray, Tensor)):
        return data.item()
    else:
        return data
