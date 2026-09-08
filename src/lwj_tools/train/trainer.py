"""手写 :class:`Trainer` ABC。

把训练循环 / checkpoint / 优化器钩子委托给 :mod:`lwj_tools.train.loops` 与
:mod:`lwj_tools.train.checkpoint` 的自由函数，子类只需关注数据加载与评估。

子类必须实现的抽象方法：

- :meth:`build_train_loader` —— 构建 train dataloader
- :meth:`build_val_loader` —— 构建 val dataloader
- :meth:`evaluate_model` —— 在给定 dataloader 上评估，返回 dict 指标

可选覆写的钩子（在 :mod:`.loops` 中默认无操作）：

- :meth:`before_optim_lr_scheduler` / :meth:`optim_lr_scheduler` / :meth:`after_optim_lr_scheduler`

> 推荐用 Transformers 的 Trainer 和 TrainingArguments；本 Trainer 主要用于
> (1) 自定义 ``build_*_loader`` (2) val_file_path 不是必需的；
> 2-1) 如果不设置，模型会一直训练到最后
> 2-2) 如果你设置了 val_file_path 并重写了 :meth:`evaluate_model`，则可以根据
>      你设置的 golden metric 来判断模型质量并触发 checkpoint 保存

Example:
    >>> from lwj_tools.train.trainer import Trainer
    >>> from lwj_tools.train.optimizer import TrainingArguments
    >>> from torch.utils.data import DataLoader
    >>> class MyTrainer(Trainer):
    ...     def build_train_loader(self, path):
    ...         return DataLoader([{"x": 1, "y": 0}] * 32, batch_size=8)
    ...     def build_val_loader(self, path):
    ...         return self.build_train_loader(path)
    ...     def evaluate_model(self, dataloader):
    ...         return {"loss": 0.1, "acc": 0.9}
    >>> import torch.nn as nn
    >>> trainer = MyTrainer(
    ...     model=nn.Linear(1, 1),
    ...     train_file_path="train.jsonl",
    ...     val_file_path="val.jsonl",
    ...     config=TrainingArguments(total_steps=10, epochs=1, device="cpu"),
    ... )
    >>> trainer.train()
"""
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
from transformers import set_seed

from ..common.files import clean_dir
from ..common.logging import get_logger
from ..common._typing import FilePath
from ..io.writer import FileWriter
from . import checkpoint as _checkpoint
from . import loops as _loops
from .checkpoint import MODEL, OPTIMIZER, SCHEDULER, TRAIN_STATE
from .optimizer import (
    OPTIM_CLS_MAP,
    SCHEDULER_CLS_MAP,
    Stage,
    TrainingArguments,
    build_optimizer,
    build_scheduler,
)

LOGGER = get_logger("lwj_tools")


class Trainer(ABC):
    """训练器抽象基类。

    类属性 :attr:`MODEL` / :attr:`OPTIMIZER` / :attr:`SCHEDULER` / :attr:`TRAIN_STATE`
    是 checkpoint 文件名（不带扩展名），与 :mod:`.checkpoint` 中的模块级常量指向
    同一字符串值 —— 保留 ``self.MODEL`` 等访问语义，方便子类扩展。
    """

    MODEL = MODEL
    OPTIMIZER = OPTIMIZER
    SCHEDULER = SCHEDULER
    TRAIN_STATE = TRAIN_STATE

    def __init__(
        self,
        model: nn.Module,
        train_file_path: FilePath,
        config: TrainingArguments = None,
        val_file_path: Optional[FilePath] = None,
        test_file_path: Optional[FilePath] = None,
    ):
        """初始化训练器。

        Args:
            model: 待训练模型。
            train_file_path: 训练数据文件路径。
            config: :class:`TrainingArguments`；省略时使用默认值。
            val_file_path: 验证数据文件路径；省略时不评估 / 一直跑到 ``total_steps``。
            test_file_path: 测试数据文件路径；省略时不进行 test 评估。
        """
        if config is None:
            config = TrainingArguments()

        self._config = config
        assert self._config.seed >= 0, "seed must be greater than or equal to 0."
        set_seed(self._config.seed)

        if self._config.overwrite_output_dir:
            clean_dir(self._config.output_dir)

        # 构建 dataloader
        self._train_loader = self.build_train_loader(train_file_path)
        self._val_loader = (
            self.build_val_loader(val_file_path) if val_file_path else None
        )
        self._test_loader = (
            self.build_test_loader(test_file_path) if test_file_path else None
        )

        self._check_and_set_default_config()

        # 构建 optimizer / scheduler
        self._model = model.to(self._config.device)
        self._optimizer = build_optimizer(
            optimizer_name=self._config.optimizer_name,
            optimizer_specific_kwargs=self._config.optimizer_specific_kwargs,
            trainable_params=self.get_model_trainable_params(),
        )
        self._scheduler = (
            build_scheduler(
                scheduler_name=self._config.scheduler_name,
                optimizer=self._optimizer,
                num_warmup_steps=self._config.warmup_steps,
                num_training_steps=self._config.total_steps,
                scheduler_specific_kwargs=self._config.scheduler_specific_kwargs,
            )
            if self._config.warmup_steps > 0
            else None
        )

        # 初始化训练状态
        # global_steps = steps * gradient_accumulation_steps
        self._train_states = {
            "global_steps": 0,  # 累计迭代次数
            "steps": 0,  # 累计 optimizer.step 次数
            "accumulate_loss": 0,
            "patience": self._config.patience,
            "best_ckpt_paths": [],
            "best_golden_metric_value": (
                float("inf") if self._config.lower_is_better else float("-inf")
            ),
        }

        if self._config.resume_checkpoint_path:
            _checkpoint.load_checkpoint(self)

        # 配置 logger
        if self._config.verbose:
            os.makedirs(self._config.output_dir, exist_ok=True)
            self._train_logger = open(
                os.path.join(self._config.output_dir, "train_log.jsonl"),
                encoding="utf-8",
                mode="a+",
                buffering=1,
            )
            if self._val_loader:
                self._eval_logger = open(
                    os.path.join(self._config.output_dir, "eval_log.jsonl"),
                    encoding="utf-8",
                    mode="a+",
                    buffering=1,
                )
            if self._config.adopt_tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(
                    log_dir=os.path.join(self._config.output_dir),
                )

        # 落盘 config.yaml
        FileWriter.dump(
            asdict(self._config),
            os.path.join(self._config.output_dir, "config.yaml"),
        )

    @property
    def model(self) -> nn.Module:
        """当前训练中的模型。"""
        return self._model

    @property
    def config(self) -> TrainingArguments:
        """训练配置。"""
        return self._config

    @property
    def train_states(self) -> dict:
        """训练状态字典（含 ``global_steps`` / ``steps`` / ``best_ckpt_paths`` 等）。"""
        return self._train_states

    def _check_and_set_default_config(self):
        """规范化配置字段、补算派生量（``total_steps`` / ``warmup_steps``），并做 sanity assert。"""
        self._config.optimizer_name = self._config.optimizer_name.lower()
        self._config.scheduler_name = self._config.scheduler_name.lower()
        if "cuda" in self._config.device and not torch.cuda.is_available():
            LOGGER.warning(f"CUDA is not available, using CPU instead.")
            self._config.device = "cpu"

        assert (
                self._config.train_batch_size > 0
        ), "train_batch_size must be greater than 0."
        assert (
                self._config.eval_batch_size > 0
        ), "eval_batch_size must be greater than 0."

        if self._config.resume_checkpoint_path:
            assert os.path.exists(
                self._config.resume_checkpoint_path,
            ), f"{self._config.resume_checkpoint_path} does not exist."

        assert self._config.epochs > 0, "epochs must be greater than 0."
        assert (
                self._config.optimizer_name in OPTIM_CLS_MAP
        ), f"{self._config.optimizer_name} not supported yet."
        assert self._config.learning_rate > 0, "learning_rate must be greater than 0."
        assert (
                self._config.scheduler_name in SCHEDULER_CLS_MAP
        ), f"{self._config.scheduler_name} not supported yet."
        assert (
                0 <= self._config.warmup_ratio <= 1
        ), "warmup_ratio must be between 0 and 1."
        assert (
                0 <= self._config.warmup_steps
        ), "warmup_steps must be greater than or equal to 0."
        assert (
                self._config.gradient_accumulation_steps > 0
        ), "gradient_accumulation_steps must be greater than 0."

        warmup_ratio = self._config.warmup_ratio
        warmup_steps = self._config.warmup_steps

        self._config.total_steps = (
                int(len(self._train_loader) / self._config.gradient_accumulation_steps)
                * self._config.epochs
        )
        if warmup_steps != 0:
            if warmup_ratio != 0:
                LOGGER.warning(
                    f"warmup_steps and warmup_ratio are both set, warmup_ratio will be ignored.",
                )
            self._config.warmup_ratio = warmup_steps / self._config.total_steps
        elif warmup_ratio != 0:
            self._config.warmup_steps = int(self._config.total_steps * warmup_ratio)

        assert (
                self._config.save_total_limit > 0
        ), "save_total_limit must be greater than 0."

        if self._val_loader is not None:
            if isinstance(self._config.eval_steps, str):
                self._config.eval_steps = int(
                    len(self._train_loader) / self._config.gradient_accumulation_steps,
                )
            assert self._config.eval_steps > 0, "eval_steps must be greater than 0."

        if self._test_loader is not None:
            self._config.eval_best_model_on_test = True

    @abstractmethod
    def build_train_loader(self, train_file_path: FilePath):
        """构建 train dataloader —— 子类必须实现。"""
        raise NotImplementedError

    @abstractmethod
    def build_val_loader(self, val_file_path: FilePath):
        """构建 val dataloader —— 子类必须实现。"""
        raise NotImplementedError

    def build_test_loader(self, test_file_path: FilePath):
        """构建 test dataloader —— 默认复用 :meth:`build_val_loader`。"""
        return self.build_val_loader(test_file_path)

    def get_model_trainable_params(self):
        """收集 ``requires_grad=True`` 的参数（optimizer 优化范围）。"""
        return list(filter(lambda p: p.requires_grad, self._model.parameters()))

    def _log(self, result_dict: dict, stage: Stage, custom_logger=None):
        """把 ``result_dict`` 中 ``log_items`` 包含的字段按阶段写到对应 logger。"""
        from ..nn.helpers import convert_data_to_normal_type  # 局部 import

        if stage == Stage.TRAIN:
            log_items = self._config.train_log_items
            fp = self._train_logger
        elif stage == Stage.EVAL:
            log_items = self._config.eval_log_items
            fp = self._eval_logger
        else:
            log_items = self._config.test_log_items
            fp = None

        if custom_logger:
            fp = custom_logger

        steps = self._train_states["steps"]
        log_save_dict = {"steps": steps}
        for k, v in result_dict.items():
            if k in log_items:
                v = convert_data_to_normal_type(v)
                log_save_dict[k] = v
                if self._config.adopt_tensorboard:
                    if isinstance(v, (float, int)):
                        self._tb_writer.add_scalar(f"{stage}/{k}", v, steps)

        if fp:
            fp.write(json.dumps(log_save_dict, ensure_ascii=False) + "\n")
            fp.flush()

    @abstractmethod
    def evaluate_model(self, dataloader):
        """在 ``dataloader`` 上评估，返回 dict 指标 —— 子类必须实现。"""
        raise NotImplementedError

    def evaluate_model_on_test(self):
        """对所有 best checkpoint 在 test loader 上评估 —— 委托给 :func:`loops.evaluate_model_on_test`。"""
        return _loops.evaluate_model_on_test(self)

    def model_forward(self, batch) -> dict:
        """单 batch forward —— 委托给 :func:`loops.model_forward`。"""
        return _loops.model_forward(self, batch)

    def before_optim_lr_scheduler(self):
        """优化器 step 之前的钩子 —— 委托给 :func:`loops.before_optim_lr_scheduler`。"""
        return _loops.before_optim_lr_scheduler(self)

    def optim_lr_scheduler(self):
        """优化器 step —— 委托给 :func:`loops.optim_lr_scheduler`。"""
        return _loops.optim_lr_scheduler(self)

    def after_optim_lr_scheduler(self):
        """优化器 step 之后的钩子 —— 委托给 :func:`loops.after_optim_lr_scheduler`。"""
        return _loops.after_optim_lr_scheduler(self)

    def train(self):
        """启动主训练循环 —— 委托给 :func:`loops.run_train_loop`。"""
        return _loops.run_train_loop(self)
