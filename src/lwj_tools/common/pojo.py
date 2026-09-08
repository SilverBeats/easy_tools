"""Dict-like 混入基类与 dataclass 兼容性工具。

:class:`DictLike` 通过 :attr:`object.__dict__` 把实例字段暴露成 dict 风格接口，
便于以 ``obj["key"]`` 访问字段、``obj.to_dict()`` 序列化为普通字典。

设计要点：

- 默认实现用 :meth:`object.__dict__`，因此与普通 ``@dataclass`` 兼容。
- 若子类使用 ``dataclass(slots=True)`` 等不创建 ``__dict__`` 的存储策略，
  可覆写 :meth:`DictLike._fields` 改用 :func:`dataclasses.asdict` 等其他来源。
- 嵌套的 :class:`DictLike` 字段在 :meth:`DictLike.to_dict(deep=True)` 时会
  递归调用各自的 :meth:`to_dict`。

Example:
    >>> from lwj_tools.common.pojo import DictLike
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class User(DictLike):
    ...     name: str
    ...     age: int
    >>> u = User(name="alice", age=30)
    >>> u["name"]
    'alice'
    >>> u.to_dict()
    {'name': 'alice', 'age': 30}
    >>> "age" in u
    True
"""
from typing import Any, Iterator, List, Optional


class DictLike:
    """Dict-like 混入基类。

    通过 :attr:`object.__dict__` 暴露实例字段并提供 ``__getitem__`` / ``__setitem__`` /
    ``__contains__`` / ``to_dict`` / ``update`` 等 dict 风格方法。

    子类可在 :meth:`_fields` 中返回自定义字段来源（如 ``dataclass(slots=True)`` 时）。
    """

    def _fields(self) -> dict:
        """返回当前实例的字段映射。

        默认使用 :attr:`object.__dict__`；子类使用其他存储策略时可覆写。
        """
        return self.__dict__

    def __getitem__(self, key: str) -> Any:
        fields = self._fields()
        if key not in fields:
            raise KeyError(key)
        return fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._fields()[key] = value

    def __delitem__(self, key: str) -> None:
        fields = self._fields()
        if key not in fields:
            raise KeyError(key)
        del fields[key]

    def __contains__(self, key: object) -> bool:
        return key in self._fields()

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields())

    def __len__(self) -> int:
        return len(self._fields())

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """``dict.get`` 等价：缺键返回 ``default``，无 ``default`` 时为 ``None``。"""
        return self._fields().get(key, default)

    def keys(self):
        """``dict.keys`` 等价：返回实例字段名的 :class:`dict_keys` 视图。"""
        return self._fields().keys()

    def values(self):
        """``dict.values`` 等价：返回实例字段值的 :class:`dict_values` 视图。"""
        return self._fields().values()

    def items(self):
        """``dict.items`` 等价：返回 ``(field, value)`` 的 :class:`dict_items` 视图。"""
        return self._fields().items()

    def to_dict(self, deep: bool = True) -> dict:
        """把实例字段序列化为普通 :class:`dict`。

        Args:
            deep: 为 ``True`` 时嵌套 :class:`DictLike` 字段会递归 ``to_dict()``；
                为 ``False`` 时仅浅拷贝。

        Returns:
            序列化后的字典。
        """
        if deep:
            return {
                k: v.to_dict() if isinstance(v, DictLike) else v
                for k, v in self.items()
            }
        return dict(self._fields())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self._fields() == other
        if isinstance(other, DictLike):
            return self._fields() == other._fields()
        return NotImplemented

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self._fields().items())
        return f"{type(self).__name__}({body})"

    def update(self, _dict: dict, skip_keys: Optional[List[str]] = None):
        """把 ``_dict`` 中的键值对 ``setattr`` 到本实例，跳过 ``skip_keys`` 列出的字段。"""
        skip_keys = skip_keys or []
        for k, v in _dict.items():
            if k in skip_keys:
                continue
            setattr(self, k, v)
