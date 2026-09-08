#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用工具集合，按职责拆分为 8 个子模块：

- :mod:`._typing` 跨子包共享的类型别名（:data:`FilePath`）
- :mod:`.random`  随机抽样与洗牌
- :mod:`.strings` 字符串转换（argparse 布尔、驼峰转蛇形）
- :mod:`.ids`     UUID / MD5 / Base64 / URL 校验
- :mod:`.files`   文件与目录操作、样本去重
- :mod:`.math`    余弦相似度、GloVe 加载
- :mod:`.logging` 日志器创建
- :mod:`.pojo`    :class:`DictLike` 混入基类

调用方可从子模块精细导入（推荐）::

    from lwj_tools.common.files import get_file_name_and_ext
    from lwj_tools.common.logging import get_logger

也可从包顶层批量导入（保留所有公开函数名）::

    from lwj_tools.common import get_logger, get_uuid, rm_file
"""

from ._typing import FilePath
from .files import (
    clean_dir,
    get_dir_file_path,
    get_file_name_and_ext,
    get_unprocessed_samples,
    load_glove,
    rm_dir,
    rm_file,
)
from .ids import get_base64, get_md5_id, get_uuid, is_url
from .logging import get_logger
from .math import cosine_similarity
from .random import random_choice, shuffle
from .strings import camel_to_snake, str2bool

__all__ = [
    # random
    "random_choice",
    "shuffle",
    # strings
    "str2bool",
    "camel_to_snake",
    # ids
    "get_uuid",
    "get_md5_id",
    "get_base64",
    "is_url",
    # files
    "get_file_name_and_ext",
    "get_dir_file_path",
    "rm_file",
    "rm_dir",
    "clean_dir",
    "get_unprocessed_samples",
    "load_glove",
    # math
    "cosine_similarity",
    # logging
    "get_logger",
    # typing
    "FilePath",
]
