#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""随机抽样与洗牌工具。"""
import random
from typing import Any, List, Sequence, Set, Union


def random_choice(arr: Union[Sequence, Set], n: int = 1) -> List[Any]:
    """从数组中随机选择n个元素

    Args:
        arr: 数组
        n: 随机选择的元素个数

    Returns:
        List[Any]: 随机选择的元素
    """
    return random.sample(arr, min(n, len(arr)))


def shuffle(arr: List[Any], n: int = 1):
    """随机打乱数组中的元素顺序（原地操作）

    Args:
        arr: 数组
        n: 随机打乱的次数
    """
    for _ in range(n):
        random.shuffle(arr)
