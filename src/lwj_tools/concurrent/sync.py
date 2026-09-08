#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步并发运行器，基于 :mod:`concurrent.futures`。

通过传入不同的 ``executor_cls``（:class:`~concurrent.futures.ProcessPoolExecutor`
或 :class:`~concurrent.futures.thread.ThreadPoolExecutor`）得到进程级或线程级并发。
两个具体子类 :class:`MultiProcessRunner` / :class:`MultiThreadingRunner` 已预设好
对应执行器，一般直接使用即可。

调用方实现的 :code:`worker_func` 形如::

    def worker(idx, sample):
        do_something(sample)
        return sample["result"]

异步版见 :mod:`lwj_tools.concurrent.async_runner`。
"""
import os
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import Future, as_completed
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from ..common import get_logger
from ..errors import ConcurrentError

LOGGER = get_logger("lwj_tools")


class ConcurrentRunner(ABC):
    """基于 :mod:`concurrent.futures` 的同步并发运行器基类。

    通过 :attr:`executor_cls` 选择进程池或线程池；子任务签名必须是
    ``worker_func(idx, sample)``，返回值经 :meth:`finished_func` 二次加工后填入
    结果列表。失败任务的返回值是 :class:`ConcurrentError` 实例（含 traceback）。

    Attributes:
        executor_cls: 实际的执行器类（:class:`ProcessPoolExecutor` 或
            :class:`ThreadPoolExecutor`）。
        num_workers: 最大并发数。
        use_pbar: 是否显示 :mod:`tqdm` 进度条。
        stop_by_error: 首个失败任务是否立即取消其他任务并抛错。
        verbose: 是否对失败任务打印 traceback 日志。
        need_order: 结果是否按 ``samples`` 顺序排列。
    """

    def __init__(
        self,
        executor_cls: Callable,
        num_workers: int = -1,
        use_pbar: bool = True,
        stop_by_error: bool = False,
        verbose: bool = False,
        need_order: bool = False,
    ):
        """初始化并发运行器。

        Args:
            executor_cls: 执行器类，如 :class:`ProcessPoolExecutor` /
                :class:`ThreadPoolExecutor`。
            num_workers: 最大并发数；``-1`` 表示 :func:`os.cpu_count`。
            use_pbar: 是否显示 :mod:`tqdm` 进度条。
            stop_by_error: 首个失败任务是否取消其余并抛出
                :class:`ConcurrentError`。
            verbose: 是否对失败任务记录日志。
            need_order: 是否按输入顺序返回结果。
        """
        if num_workers == -1:
            num_workers = os.cpu_count() or 1
        self.executor_cls = executor_cls
        self.num_workers = num_workers
        self.use_pbar = use_pbar
        self.stop_by_error = stop_by_error
        self.verbose = verbose
        self.need_order = need_order

    @abstractmethod
    def worker_func(self, idx: int, sample: Any):
        """子任务函数。子类必须实现，返回任意可序列化结果。

        Args:
            idx: 当前 sample 在输入列表中的下标。
            sample: 当前任务数据。

        Returns:
            任意可序列化对象，作为任务结果返回。
        """
        raise NotImplementedError

    def finished_func(self, idx: int, result: Any):
        """对每个任务结果做后处理，默认直接返回。子类可覆写。

        Args:
            idx: 当前 sample 的下标。
            result: :meth:`worker_func` 返回的原始结果。

        Returns:
            后处理后的最终结果。
        """
        return result

    def __call__(
        self,
        samples: List[Any],
        n_samples: Optional[int] = None,
        pbar_desc: str = "Running",
    ) -> List[Any]:
        """并行执行所有 sample 并按完成顺序（或输入顺序）收集结果。

        Args:
            samples: 任务列表，每项是一个样本。
            n_samples: 截取前 N 个；``None`` 表示全部。
            pbar_desc: 进度条描述。

        Returns:
            与 ``samples`` 等长的结果列表。失败的位置填
            :class:`ConcurrentError`。
        """
        if n_samples is None:
            n_samples = len(samples)
        else:
            samples = samples[:n_samples]

        results: List[Any] = [None] * n_samples
        pbar = None
        if self.use_pbar:
            pbar = tqdm(total=n_samples, desc=pbar_desc, dynamic_ncols=True, leave=True)

        with self.executor_cls(max_workers=min(n_samples, self.num_workers)) as executor:
            try:
                future_to_idx: Dict[Future, int] = {}
                for idx, sample in enumerate(samples):
                    task = executor.submit(self.worker_func, idx, sample)
                    future_to_idx[task] = idx

                for task in as_completed(future_to_idx):
                    idx = future_to_idx[task]
                    try:
                        result = task.result()
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
                            for t in future_to_idx:
                                if not t.done():
                                    t.cancel()
                            executor.shutdown(wait=False)
                            raise error
                    finally:
                        if pbar:
                            pbar.update(1)
                            pbar.refresh()
            finally:
                if pbar:
                    pbar.close()
        return results


class MultiProcessRunner(ConcurrentRunner):
    """多进程并发运行器（基于 :class:`ProcessPoolExecutor`）。

    适用于 CPU 密集型任务；注意 worker 函数必须是可 pickle 的（模块级定义）。
    """

    def __init__(
        self,
        num_workers: int = -1,
        use_pbar: bool = True,
        stop_by_error: bool = False,
        verbose: bool = False,
        need_order: bool = False,
    ):
        super().__init__(
            ProcessPoolExecutor,
            num_workers,
            use_pbar,
            stop_by_error,
            verbose,
            need_order,
        )


class MultiThreadingRunner(ConcurrentRunner):
    """多线程并发运行器（基于 :class:`ThreadPoolExecutor`）。

    适用于 I/O 密集型任务；线程间共享内存，但 GIL 限制下 CPU 任务无法真正并行。
    """

    def __init__(
        self,
        num_workers: int = -1,
        use_pbar: bool = True,
        stop_by_error: bool = False,
        verbose: bool = False,
        need_order: bool = False,
    ):
        super().__init__(
            ThreadPoolExecutor,
            num_workers,
            use_pbar,
            stop_by_error,
            verbose,
            need_order,
        )
