#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyTorch 模型辅助工具：参数量统计、冻结/解冻、模块克隆、数据搬运、类型规整。"""
from copy import deepcopy
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def calc_model_params(model: nn.Module) -> int:
    """计算模型可训练参数量

    Args:
        model: 模型

    Returns:
        int: 模型可训练参数量
    """
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = int(sum([np.prod(p.size()) for p in model_parameters]))
    return total_params


def freeze_model(model: nn.Module, skip_param_names: Optional[List[str]] = None):
    """冻结模型参数

    Args:
        model: 模型
        skip_param_names: 要跳过的参数名
    """
    if skip_param_names is None:
        skip_param_names = []
    for name, param in model.named_parameters():
        if any(s_p in name for s_p in skip_param_names):
            continue
        param.requires_grad = False


def unfreeze_model(model: nn.Module, skip_param_names: Optional[List[str]] = None):
    """模型参数解冻

    Args:
        model: 模型
        skip_param_names: 要跳过的参数名
    """
    if skip_param_names is None:
        skip_param_names = []
    for name, param in model.named_parameters():
        if any(s_p in name for s_p in skip_param_names):
            continue
        param.requires_grad = True


def clone_module(module: nn.Module, n: int) -> List[nn.Module]:
    """复制模块

    Args:
        module: 模块
        n: 复制的次数

    Returns:
        List[nn.Module]: 复制的模块
    """
    return [deepcopy(module) for _ in range(n)]


def data_2_device(data: Any, device: Union[torch.device, str]) -> Any:
    """ 尝试将数据（可嵌套）移动到指定的设备

    Args:
        data: 数据
        device: 设备

    Returns:
        Any: 移动到指定设备后的数据
    """
    if isinstance(data, dict):
        return {k: data_2_device(v, device) for k, v in data.items()}
    elif isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [data_2_device(item, device) for item in data]
    else:
        return data


def convert_data_to_normal_type(data: Any) -> Any:
    """尝试将数据（可嵌套）转换为普通的 int float 类型

    Args:
        data: 数据

    Returns:
        Any: 转换为普通类型的数据
    """
    if isinstance(data, (list, tuple)):
        return [convert_data_to_normal_type(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_data_to_normal_type(v) for k, v in data.items()}
    elif isinstance(data, (np.ndarray, Tensor)):
        return data.item()
    else:
        return data
