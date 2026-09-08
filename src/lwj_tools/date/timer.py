#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计时工具：装饰器与上下文管理器两种用法。

- :class:`Timer` —— 上下文管理器形式，适合手写代码块计时。
- :func:`timecost` —— 装饰器形式，适合给某个函数整体计时，并把返回值与耗时
  一并包装在 :class:`TimeProxyResult` 中返回。

时间单位可选 ``s`` / ``ms`` / ``ns``；底层计时函数默认为
:func:`time.perf_counter`（单调时钟，适合测量间隔）。

Example:
    >>> from lwj_tools.date.timer import Timer, timecost
    >>>
    >>> with Timer(unit="ms") as t:
    ...     heavy_work()
    >>> print(t.elapsed)  # ms
    >>>
    >>> @timecost(unit="ms")
    ... def compute():
    ...     return 42
    >>> r = compute()
    >>> r.result, r.timecost
    (42, ...)
"""
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass
class TimeProxyResult:
    """装饰器 :func:`timecost` 的返回值包装。

    Attributes:
        result: 被装饰函数的原始返回值。
        timecost: 函数执行耗时（单位由 ``unit`` 决定）。
    """

    result: Optional[Any] = None
    timecost: float = 0  # s


def timecost(
    proxy_func, time_func=time.perf_counter, unit: Literal["s", "ms", "ns"] = "s"
):
    """装饰器：在调用被装饰函数前后计时，返回 :class:`TimeProxyResult`。

    Args:
        proxy_func: 被装饰的函数。
        time_func: 计时函数，默认 :func:`time.perf_counter`。
        unit: 时间单位，``"s"`` / ``"ms"`` / ``"ns"``，默认 ``"s"``。

    Returns:
        :class:`TimeProxyResult`，含 ``result`` 与 ``timecost`` 两字段。

    Example:
        >>> @timecost
        ... def foo():
        ...     time.sleep(0.01)
        ...     return 1
        >>> foo()
        TimeProxyResult(result=1, timecost=...)
    """

    def wrapper(*args, **kwargs) -> TimeProxyResult:
        with Timer(time_func, unit) as timer:
            func_result = proxy_func(*args, **kwargs)
        return TimeProxyResult(func_result, timer.elapsed)

    return wrapper


class Timer:
    """上下文管理器式计时器。

    用法::

        with Timer(unit="ms") as t:
            ...
        print(t.elapsed)

    Attributes:
        elapsed: 累计耗时（按 ``unit`` 换算后）。
        is_running: 是否正在计时（属性）。
    """

    def __init__(self, func=time.perf_counter, unit: Literal["s", "ms", "ns"] = "s"):
        """初始化计时器。

        Args:
            func: 计时函数，默认 :func:`time.perf_counter`。
            unit: 时间单位，``"s"`` / ``"ms"`` / ``"ns"``。
        """
        self.elapsed = 0.0
        self._func = func
        self._start = None
        self._unit = unit
        self._unit_factor = {"s": 1, "ms": 1e3, "ns": 1e6}.get(unit, 1)

    def start(self):
        """启动计时；若已启动则抛 :class:`RuntimeError`。

        Raises:
            RuntimeError: 重复调用 ``start()``。
        """
        if self._start is not None:
            raise RuntimeError("Already started")
        self._start = self._func()

    def stop(self):
        """停止计时并把本段耗时累加到 :attr:`elapsed`；未启动则抛 :class:`RuntimeError`。

        Raises:
            RuntimeError: 未调用 ``start()`` 就调用 ``stop()``。
        """
        if self._start is None:
            raise RuntimeError("Not started")
        end = self._func()
        self.elapsed += (end - self._start) * self._unit_factor
        self._start = None

    def reset(self):
        """把 :attr:`elapsed` 清零（不改变运行状态）。"""
        self.elapsed = 0.0

    @property
    def is_running(self) -> bool:
        """是否正在计时（``start`` 后未 ``stop``）。"""
        return self._start is not None

    def __enter__(self):
        """进入 ``with`` 块时自动 :meth:`start` 并返回自身。"""
        self.start()
        return self

    def __exit__(self, *args):
        """退出 ``with`` 块时自动 :meth:`stop`。"""
        self.stop()
