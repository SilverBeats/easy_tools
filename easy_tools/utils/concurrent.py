#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import traceback
from abc import ABC
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Iterable, Optional

from tqdm import tqdm

from .constant import LOGGER
from ..errors import ConcurrentError

__all__ = ["MultiProcessRunner", "MultiThreadingRunner"]


def wrapper(
    idx: int,
    sample: Any,
    worker_func: Callable,
    callback_func: Optional[Callable] = None,
):
    result = worker_func(sample)
    if callback_func:
        result = callback_func(result, sample)
    return idx, result


class ConcurrentRunner(ABC):
    def __init__(
        self,
        executor_cls: Callable,
        num_workers: int = -1,
        use_pbar: bool = True,
        stop_by_error: bool = False,
        verbose: bool = False,
        need_order: bool = False,
    ):
        if num_workers == -1:
            num_workers = os.cpu_count()
        self.executor_cls = executor_cls
        self.num_workers = num_workers
        self.use_pbar = use_pbar
        self.stop_by_error = stop_by_error
        self.verbose = verbose
        self.need_order = need_order

    def __call__(
        self,
        samples: Iterable,
        worker_func: Callable,
        callback_func: Optional[Callable] = None,
        finish_func: Optional[Callable] = None,
        n_samples: int = None,
        desc: str = "Running",
    ):
        if n_samples is None:
            try:
                n_samples = len(samples)
            except TypeError:
                samples = list(samples)
                n_samples = len(samples)

        wrapper_func = partial(
            wrapper,
            worker_func=worker_func,
            callback_func=callback_func,
        )
        results = [None] * n_samples if self.need_order else []
        with self.executor_cls(
            max_workers=min(n_samples, self.num_workers),
        ) as executor:
            try:
                tasks = [
                    executor.submit(wrapper_func, idx, sample)
                    for idx, sample in tqdm(
                        enumerate(samples),
                        total=n_samples,
                        dynamic_ncols=True,
                        leave=False,
                        desc="Submitting Task",
                    )
                ]
            except Exception:
                error = ConcurrentError(traceback.format_exc())
                if self.verbose:
                    LOGGER.error(error)
                if self.stop_by_error:
                    raise error

            pbar = None
            if self.use_pbar:
                pbar = tqdm(total=n_samples, desc=desc, dynamic_ncols=True, leave=True)

            try:
                for task in as_completed(tasks):
                    try:
                        idx, result = task.result()
                        if finish_func:
                            finish_func(idx, result)
                        if self.need_order:
                            results[idx] = result
                        else:
                            results.append(result)
                    except Exception:
                        error = ConcurrentError(traceback.format_exc())
                        if self.verbose:
                            LOGGER.error(error)
                        if self.stop_by_error:
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
