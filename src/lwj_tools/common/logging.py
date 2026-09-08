#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日志工具。"""
import logging
import os
from typing import Optional


def get_logger(
    name: str,
    level: str = "info",
    formatter: Optional[str] = None,
    log_path: Optional[str] = None,
) -> logging.Logger:
    """获取 logger

    Args:
        name: logger 名称
        level: log 级别
        formatter: log formatter
        log_path: log 文件路径
    """
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }

    assert level in LEVELS

    logger = logging.getLogger(name)

    if not logger.handlers:
        level_ = LEVELS[level]
        logger.setLevel(level_)

        fmt = (
            formatter
            if formatter is not None
            else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

        ch = logging.StreamHandler()
        ch.setLevel(level_)
        ch.setFormatter(log_formatter)
        logger.addHandler(ch)

        if log_path is not None:
            dirname = os.path.dirname(log_path)
            os.makedirs(dirname, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(level_)
            fh.setFormatter(log_formatter)
            logger.addHandler(fh)

    return logger
