#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ID 生成与编码工具：UUID、MD5、Base64、URL 校验。"""
import base64
import hashlib
import uuid
from urllib.parse import urlparse

from ._typing import FilePath


def get_uuid(prefix=None) -> str:
    """获取 uuid

    Args:
        prefix: uuid 的前缀

    Returns:
        str: uuid
    """
    if prefix is not None:
        return f"{prefix}-{uuid.uuid4().hex}"
    return uuid.uuid4().hex


def get_md5_id(text: str) -> str:
    """获取文本的MD5值

    Args:
        text: 文本

    Returns:
        str: MD5值
    """
    hash_str = hashlib.md5(text.encode("utf-8")).hexdigest()
    return hash_str


def get_base64(file_path: FilePath) -> str:
    """ 获取文件的base64编码

    Args:
        file_path: 文件路径

    Returns:
        str: base64编码
    """
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded


def is_url(url: str) -> bool:
    """判断是否为有效URL

    Args:
        url: URL字符串

    Returns:
        bool: 是否为有效URL
    """
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except Exception as e:
        return False
