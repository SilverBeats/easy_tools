#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并发运行器集合。

- 同步版（:mod:`.sync`）：基于 :mod:`concurrent.futures`，提供
  :class:`ConcurrentRunner`、:class:`MultiProcessRunner`、:class:`MultiThreadingRunner`。
- 异步版（:mod:`.async_runner`）：基于 :mod:`asyncio` + :mod:`httpx`，提供
  :class:`AsyncRunner`。
"""
from .async_runner import AsyncRunner
from .sync import ConcurrentRunner, MultiProcessRunner, MultiThreadingRunner

__all__ = [
    "AsyncRunner",
    "ConcurrentRunner",
    "MultiProcessRunner",
    "MultiThreadingRunner",
]
