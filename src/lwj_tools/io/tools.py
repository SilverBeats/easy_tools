#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IO 装饰器工具。

:func:`ext_check` 在文件读写函数被调用前检查 ``file_path`` 的扩展名是否在白名单内，
不是则抛 :class:`lwj_tools.errors.FileTypeError`；路径缺失则抛
:class:`lwj_tools.errors.FileReadError`。
"""
from typing import Callable, List, Union

from ..errors import FileReadError, FileTypeError
from ..common.files import get_file_name_and_ext


def ext_check(ext: Union[str, List[str]]):
    """检查文件扩展名

    Args:
        ext: 一个或多个文件扩展名（没有点）
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            file_path = kwargs.get("file_path", None)
            if file_path is None and len(args) > 0:
                func_name = func.__name__
                file_path_index = 0 if "read" in func_name else 1
                file_path = args[file_path_index]

            if not file_path:
                raise FileReadError(f"{file_path} is required")

            file_ext = get_file_name_and_ext(file_path, with_dot=False)[-1]

            allowed_exts = [ext] if isinstance(ext, str) else ext
            if file_ext not in allowed_exts:
                allowed_str = ", ".join(allowed_exts)
                raise FileTypeError(f"{file_path} is not a {allowed_str} file!")

            return func(*args, **kwargs)

        return wrapper

    return decorator
