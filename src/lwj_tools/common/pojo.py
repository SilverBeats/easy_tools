"""Dict-like 混入基类与 dataclass 兼容性工具。"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Iterator, Optional


class DictLike:
    """Dict-like 混入基类。

    注意：
        若子类使用 @dataclass，建议写成
        @dataclass(eq=False, repr=False)
        否则 dataclass 自动生成的 __eq__ / __repr__ 会覆盖本类实现。
    """

    __slots__ = ()

    _SCALAR_TYPES = (type(None), bool, int, float, str, bytes)

    def _fields(self) -> dict:
        """返回当前实例的字段映射。

        默认使用 object.__dict__。
        若子类使用 dataclass(slots=True) 且没有 __dict__，
        则回退为 dataclass 字段快照。
        """
        try:
            return self.__dict__
        except AttributeError:
            if dataclasses.is_dataclass(self) and not isinstance(self, type):
                return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
            raise

    def __getitem__(self, key: str) -> Any:
        fields = self._fields()
        if key not in fields:
            raise KeyError(key)
        return fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        try:
            fields = self.__dict__
        except AttributeError:
            setattr(self, key, value)
        else:
            fields[key] = value

    def __delitem__(self, key: str) -> None:
        try:
            fields = self.__dict__
        except AttributeError:
            try:
                delattr(self, key)
            except AttributeError as exc:
                raise KeyError(key) from exc
        else:
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
        return self._fields().get(key, default)

    def keys(self):
        return self._fields().keys()

    def values(self):
        return self._fields().values()

    def items(self):
        return self._fields().items()

    def to_dict(self, deep: bool = True, *, detect_cycles: bool = True) -> dict:
        """把实例字段序列化为普通 dict。

        Args:
            deep:
                True 时递归转换嵌套的 DictLike、dataclass、Mapping、
                list、tuple、set/frozenset。
            detect_cycles:
                True 时检测循环引用并抛出 ValueError，避免无限递归。

        Returns:
            普通 dict。tuple/set 默认转为 list，以贴近 JSON 友好结构。
        """
        if not deep:
            return dict(self._fields())

        seen = set() if detect_cycles else None
        return self._convert(self._fields(), seen)

    @classmethod
    def _convert(cls, item: Any, seen: Optional[set[int]]) -> Any:
        if isinstance(item, cls._SCALAR_TYPES):
            return item

        # 循环检测：只对可能递归的容器/对象记录当前路径。
        if seen is not None:
            obj_id = id(item)
            if obj_id in seen:
                raise ValueError(f"Circular reference detected at {item!r}")

            track = isinstance(item, (DictLike, Mapping, list, tuple, set, frozenset)) \
                or (dataclasses.is_dataclass(item) and not isinstance(item, type))

            if track:
                seen.add(obj_id)
        else:
            track = False

        try:
            if isinstance(item, DictLike):
                return {k: cls._convert(v, seen) for k, v in item._fields().items()}

            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                return {
                    f.name: cls._convert(getattr(item, f.name), seen)
                    for f in dataclasses.fields(item)
                }

            if isinstance(item, Mapping):
                return {k: cls._convert(v, seen) for k, v in item.items()}

            if isinstance(item, list):
                return [cls._convert(v, seen) for v in item]

            if isinstance(item, tuple):
                return [cls._convert(v, seen) for v in item]

            if isinstance(item, (set, frozenset)):
                return [cls._convert(v, seen) for v in item]

            return item
        finally:
            if seen is not None and track:
                seen.remove(id(item))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self._fields() == other
        if isinstance(other, DictLike):
            return self._fields() == other._fields()
        return NotImplemented

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self._fields().items())
        return f"{type(self).__name__}({body})"

    def update(
        self,
        mapping: Mapping[str, Any],
        skip_keys: Optional[Sequence[str]] = None,
    ) -> None:
        """把 mapping 中的键值对 setattr 到本实例，跳过 skip_keys。"""
        skip = set(skip_keys or ())
        for k, v in mapping.items():
            if k in skip:
                continue
            setattr(self, k, v)
