#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""异步并发运行器，基于 :mod:`asyncio` + :mod:`httpx`，复用单个
:class:`httpx.AsyncClient` 以利用连接池。

调用方实现的 :code:`worker_func` 形如::

    async def worker(client, sample):
        resp = await client.get(sample["url"], timeout=10)
        resp.raise_for_status()
        return resp.json()

同步入口 :meth:`AsyncRunner.run` 内部用 :func:`asyncio.run` 启动事件循环；
异步入口 :meth:`AsyncRunner.arun` 供已有事件循环的调用方使用。

返回结构与 :class:`ConcurrentRunner` 一致：``List[Any]``，其中失败任务的
结果是 :class:`ConcurrentError` 实例。

同步版见 :mod:`lwj_tools.concurrent.sync`。

Example:
    >>> import httpx
    >>> from lwj_tools.concurrent.async_runner import AsyncRunner
    >>>
    >>> class FetchRunner(AsyncRunner):
    ...     async def worker_func(self, idx, sample, client):
    ...         resp = await client.get(sample["url"], timeout=10)
    ...         resp.raise_for_status()
    ...         return resp.text
    >>>
    >>> runner = FetchRunner(max_concurrency=8, timeout=10)
    >>> results = runner([{"url": "https://example.com"}, ...])
"""
import asyncio
import os
import traceback
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tqdm import tqdm

from ..common import get_logger
from ..errors import ConcurrentError

LOGGER = get_logger("lwj_tools")


class AsyncRunner(ABC):
    """基于 :mod:`asyncio` + :mod:`httpx` 的异步并发运行器基类。

    通过 :class:`asyncio.Semaphore` 控制并发协程数，复用单个
    :class:`httpx.AsyncClient` 共享连接池。每个任务在抛出 ``retry_on`` 中
    列出的异常时按 ``retry_backoff`` 秒间隔重试，最多重试 ``max_retries`` 次。

    Attributes:
        max_concurrency: 最大并发协程数。
        use_pbar: 是否显示 :mod:`tqdm` 进度条。
        stop_by_error: 首个失败任务是否取消其余任务并抛错。
        verbose: 是否对失败任务记录日志。
        need_order: 结果是否按 ``samples`` 顺序排列。
        max_retries: 单任务最大尝试次数（含首次）。
        retry_backoff: 重试间隔（秒）。
        retry_on: 触发重试的异常类型元组。
    """

    def __init__(
        self,
        max_concurrency: int = -1,
        use_pbar: bool = True,
        stop_by_error: bool = False,
        verbose: bool = False,
        need_order: bool = False,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        retry_on: Tuple[type, ...] = (
            httpx.TransportError,
            httpx.HTTPStatusError,
            asyncio.TimeoutError,
        ),
        **httpx_kwargs,
    ):
        """初始化异步运行器。

        Args:
            max_concurrency: 最大并发协程数（由 :class:`asyncio.Semaphore` 控制，
                非操作系统线程/进程数）；``-1`` 表示 :func:`os.cpu_count`。
            use_pbar: 是否使用 :mod:`tqdm` 进度条。
            stop_by_error: 首个失败任务是否取消其余任务并抛出
                :class:`ConcurrentError`。
            verbose: 是否对失败任务记录日志。
            need_order: 是否按输入顺序返回结果。
            max_retries: 每个任务的最大尝试次数（含首次），``>= 1``。
            retry_backoff: 重试间隔（秒）；``0`` 表示立即重试。
            retry_on: 触发重试的异常类型元组。传空元组 ``()`` 可关闭重试。
            **httpx_kwargs: 透传给 :class:`httpx.AsyncClient` 的额外参数
                （如 ``timeout``、``proxy``、``headers`` 等）。
        """
        if max_concurrency == -1:
            max_concurrency = os.cpu_count() or 1
        assert max_concurrency is not None and max_concurrency > 0, \
            "max_concurrency must be greater than 0."
        assert max_retries >= 1, "max_retries must be >= 1."
        assert retry_backoff >= 0, "retry_backoff must be >= 0."

        self.max_concurrency = max_concurrency
        self.use_pbar = use_pbar
        self.stop_by_error = stop_by_error
        self.verbose = verbose
        self.need_order = need_order
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_on = retry_on
        self._httpx_kwargs = httpx_kwargs

    @abstractmethod
    async def worker_func(self, idx: int, sample: Any, client: httpx.AsyncClient):
        """单任务协程，子类必须实现。

        Args:
            idx: 当前 sample 下标。
            sample: 任务数据。
            client: 共享的 :class:`httpx.AsyncClient` 实例。

        Returns:
            任意可序列化对象，作为该任务结果返回。
        """
        raise NotImplementedError

    def finished_func(self, idx: int, result: Any):
        """对单任务返回值的二次加工（默认原样返回）。

        Args:
            idx: 当前 sample 下标。
            result: :meth:`worker_func` 的原始返回值。

        Returns:
            后处理结果。
        """
        return result

    async def _run_one(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        idx: int,
        sample: Any,
    ) -> Any:
        """单任务执行壳：拿信号量 + 多次重试后抛最后一次异常。

        Args:
            client: 共享 :class:`httpx.AsyncClient`。
            semaphore: 并发信号量。
            idx: 当前 sample 下标。
            sample: 任务数据。

        Returns:
            :meth:`worker_func` 的返回值。

        Raises:
            Exception: ``max_retries`` 次重试后最后一次的 ``retry_on`` 类型异常。
        """
        async with semaphore:
            last_exc: Optional[BaseException] = None
            for attempt in range(self.max_retries):
                try:
                    result = await self.worker_func(idx, sample, client)
                    return result
                except self.retry_on as e:  # type: ignore
                    last_exc = e
                    if attempt < self.max_retries - 1:
                        if self.retry_backoff > 0:
                            await asyncio.sleep(self.retry_backoff)
                        continue
            assert last_exc is not None
            raise last_exc

    async def arun(
        self,
        samples: Sequence,
        n_samples: Optional[int] = None,
        pbar_desc: str = "Running",
    ) -> List[Any]:
        """异步执行入口：在已有事件循环中运行所有任务。

        Args:
            samples: 任务列表。
            n_samples: 截取前 N 个；``None`` 表示全部。
            pbar_desc: 进度条描述。

        Returns:
            与 ``samples`` 等长的结果列表。失败任务返回 :class:`ConcurrentError`。

        Raises:
            ConcurrentError: 当 ``stop_by_error=True`` 且任一任务失败时抛出。
        """
        if n_samples is None:
            n_samples = len(samples)
        else:
            samples = samples[:n_samples]

        results: List[Any] = [None] * n_samples if self.need_order else []

        pbar = None
        if self.use_pbar:
            pbar = tqdm(total=n_samples, desc=pbar_desc, dynamic_ncols=True, leave=True)

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async with httpx.AsyncClient(**self._httpx_kwargs) as client:
            task_to_idx: Dict[asyncio.Task, int] = {}
            for idx, sample in enumerate(samples):
                task = asyncio.create_task(
                    self._run_one(client, semaphore, idx, sample),
                )
                task_to_idx[task] = idx

            pending = set(task_to_idx)
            try:
                while pending:
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for done_task in done:
                        idx = task_to_idx.pop(done_task)
                        try:
                            result = done_task.result()
                            result = self.finished_func(idx, result)
                            if self.need_order:
                                results[idx] = result
                            else:
                                results.append(result)
                        except Exception:
                            error = ConcurrentError(traceback.format_exc())
                            if self.verbose:
                                LOGGER.error(error)

                            if self.need_order:
                                results[idx] = error
                            else:
                                results.append(error)

                            if self.stop_by_error:
                                for t in pending:
                                    t.cancel()
                                if pending:
                                    await asyncio.gather(*pending, return_exceptions=True)
                                raise error
                        finally:
                            if pbar:
                                pbar.update(1)
                                pbar.refresh()
                return results
            finally:
                if pbar:
                    pbar.close()

    def run(
        self,
        samples: Sequence,
        n_samples: Optional[int] = None,
        pbar_desc: str = "Running",
    ):
        """同步入口：用 :func:`asyncio.run` 包裹 :meth:`arun` 启动事件循环。

        Args:
            samples: 任务列表。
            n_samples: 截取前 N 个。
            pbar_desc: 进度条描述。

        Returns:
            与 :meth:`arun` 一致的结果列表。
        """
        return asyncio.run(self.arun(samples, n_samples, pbar_desc))

    def __call__(
        self,
        samples: Sequence,
        n_samples: Optional[int] = None,
        pbar_desc: str = "Running",
    ):
        """``runner(samples, ...)`` 等价于 ``runner.run(samples, ...)``。"""
        return self.run(samples, n_samples, pbar_desc)
