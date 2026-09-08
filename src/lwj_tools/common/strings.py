#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""字符串处理工具（argparse 布尔转换、命名风格转换）。"""
import argparse

import regex


def str2bool(v) -> bool:
    """将字符串转换为布尔值，适用于 argparse

    Args:
        v: 输入的字符串

    Returns:
        bool: 转换后的布尔值
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("不支持的值")


def camel_to_snake(name: str) -> str:
    """将驼峰命名法转换为蛇形命名法

    Args:
        name: 驼峰命名法的字符串

    Returns:
        str: 蛇形命名法的字符串
    """
    s1 = regex.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = regex.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return s2
