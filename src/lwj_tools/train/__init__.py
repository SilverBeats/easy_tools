#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练子系统，按职责拆分为 4 个子模块：

- :mod:`.optimizer`   训练配置 (TrainingArguments) + 优化器/调度器构建
- :mod:`.checkpoint`  checkpoint 保存/加载 + 早停比较
- :mod:`.loops`       训练/评估/测试主循环 + 优化器钩子
- :mod:`.trainer`     Trainer ABC，编排上述子模块

依赖 ``torch`` + ``transformers``，通过 ``pip install ".[dl]"`` 启用。
"""
from .optimizer import (
    OPTIM_CLS_MAP,
    SCHEDULER_CLS_MAP,
    Stage,
    TrainingArguments,
    build_optimizer,
    build_scheduler,
)
from .trainer import Trainer

__all__ = [
    "Trainer",
    "TrainingArguments",
    "Stage",
    "OPTIM_CLS_MAP",
    "SCHEDULER_CLS_MAP",
    "build_optimizer",
    "build_scheduler",
]
