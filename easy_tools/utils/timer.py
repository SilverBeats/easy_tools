#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass
class TimeProxyResult:
    result: Optional[Any] = None
    timecost: float = 0  # s


def timecost(
    proxy_func, time_func=time.perf_counter, unit: Literal["s", "ms", "ns"] = "s"
):
    def wrapper(*args, **kwargs) -> TimeProxyResult:
        with Timer(time_func, unit) as timer:
            func_result = proxy_func(*args, **kwargs)
        return TimeProxyResult(func_result, timer.elapsed)

    return wrapper


class Timer:
    def __init__(self, func=time.perf_counter, unit: Literal["s", "ms", "ns"] = "s"):
        self.elapsed = 0.0
        self._func = func
        self._start = None
        self._unit = unit
        self._unit_factor = {"s": 1, "ms": 1e3, "ns": 1e6}.get(unit, 1)

    def start(self):
        if self._start is not None:
            raise RuntimeError("Already started")
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError("Not started")
        end = self._func()
        self.elapsed += (end - self._start) * self._unit_factor
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def is_running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
