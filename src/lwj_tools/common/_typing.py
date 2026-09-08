"""跨子包共享的类型别名。

集中维护常用的复合类型，避免散落在各个模块里的小 :data:`Union` 重复定义。
"""
from os import PathLike
from typing import Union


FilePath = Union[str, PathLike[str]]
"""文件路径：接受字符串或 :class:`os.PathLike`。"""
