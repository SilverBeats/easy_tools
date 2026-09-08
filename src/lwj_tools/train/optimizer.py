"""训练配置与优化器 / 调度器构建。

包含：

- :class:`TrainingArguments` —— 训练超参 dataclass
- :data:`OPTIM_CLS_MAP` / :data:`SCHEDULER_CLS_MAP` —— 优化器 / 调度器名到类的映射
- :func:`build_optimizer` / :func:`build_scheduler` —— 构建器
- :class:`Stage` —— 训练阶段枚举（TRAIN / EVAL / TEST）

依赖 ``torch`` 与 ``transformers``，通过 ``pip install ".[dl]"`` 启用。

Example:
    >>> from lwj_tools.train.optimizer import (
    ...     TrainingArguments, build_optimizer, build_scheduler,
    ... )
    >>> import torch.nn as nn
    >>> model = nn.Linear(10, 2)
    >>> args = TrainingArguments(epochs=3, total_steps=100, learning_rate=1e-4)
    >>> optimizer = build_optimizer(
    ...     args.optimizer_name, args.optimizer_specific_kwargs,
    ...     [p for p in model.parameters() if p.requires_grad],
    ... )
    >>> scheduler = build_scheduler(
    ...     args.scheduler_name, optimizer,
    ...     num_warmup_steps=10, num_training_steps=100,
    ... )
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import torch
from torch.optim.optimizer import Optimizer
from transformers import SchedulerType


@dataclass
class TrainingArguments:
    """训练超参 dataclass。

    字段比较多，分组如下：

    - 日志相关：``verbose`` / ``adopt_tensorboard`` / ``use_pbar`` /
      ``train_log_items`` / ``eval_log_items`` / ``test_log_items``
    - dataloader 相关：``train_batch_size`` / ``eval_batch_size``
    - 训练相关：``device`` / ``resume_checkpoint_path`` / ``epochs`` /
      ``total_steps`` / ``optimizer_name`` / ``optimizer_specific_kwargs`` /
      ``learning_rate`` / ``scheduler_name`` / ``scheduler_specific_kwargs`` /
      ``warmup_ratio`` / ``warmup_steps`` / ``gradient_accumulation_steps`` /
      ``loss_field`` / ``patience`` / ``golden_metric`` / ``lower_is_better`` /
      ``max_grad_norm`` / ``cache_empty_steps`` / ``eval_steps`` /
      ``eval_best_model_on_test`` / ``clear_ckpt_dir``
    - 保存相关：``save_total_limit``

    每个字段都有 ``metadata={'help': ...}``，方便配合 HuggingFace ``HfArgumentParser`` 使用。
    """

    seed: int = field(default=42, metadata={"help": "Random seed."})
    output_dir: str = field(default="output", metadata={"help": "Output directory."})
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Whether to overwrite the output directory."},
    )

    # 日志
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print training progress."},
    )
    adopt_tensorboard: bool = field(
        default=False,
        metadata={"help": "Whether to use tensorboard."},
    )
    use_pbar: bool = field(
        default=True,
        metadata={"help": "Whether to use progress bar."},
    )
    train_log_items: list = field(
        default_factory=list,
        metadata={"help": "Log items during training."},
    )
    eval_log_items: list = field(
        default_factory=list,
        metadata={"help": "Log items during evaluation."},
    )
    test_log_items: list = field(
        default_factory=list,
        metadata={"help": "Log items during test."},
    )

    # dataloader
    train_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for training."},
    )
    eval_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for evaluation."},
    )

    # 训练
    device: str = field(default="cpu", metadata={"help": "Device to use."})
    resume_checkpoint_path: str = field(
        default=None,
        metadata={"help": "Path to checkpoint to resume training from."},
    )
    epochs: int = field(default=1, metadata={"help": "Number of epochs to train."})
    total_steps: int = field(
        default=0,
        metadata={"help": "Total number of training steps to perform."},
    )
    optimizer_name: str = field(default="adamw", metadata={"help": "Optimizer name."})
    optimizer_specific_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Optimizer specific kwargs."},
    )
    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate."})
    scheduler_name: str = field(default="linear", metadata={"help": "Scheduler name."})
    scheduler_specific_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Scheduler specific kwargs."},
    )
    warmup_ratio: float = field(default=0, metadata={"help": "Warmup ratio."})
    warmup_steps: int = field(default=0, metadata={"help": "Number of warmup steps."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to accumulate gradients."},
    )
    loss_field: str = field(default="loss", metadata={"help": "Loss key."})
    patience: int = field(
        default=0,
        metadata={"help": "Patience. default=0 means not adopt patience."},
    )
    golden_metric: str = field(default="loss", metadata={"help": "Golden metric."})
    lower_is_better: bool = field(
        default=True,
        metadata={"help": "Whether the lower loss is better."},
    )
    max_grad_norm: float = field(
        default=-1,
        metadata={
            "help": "Max gradient norm. If the value smaller than 0, that mean don't use grad norm"
        },
    )
    cache_empty_steps: int = field(
        default=20,
        metadata={"help": "Number of steps to clear torch.cache"},
    )
    eval_steps: Union[str, int] = field(
        default="epoch",
        metadata={"help": "Number of steps to evaluate."},
    )
    eval_best_model_on_test: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate on test set."},
    )
    clear_ckpt_dir: bool = field(
        default=True,
        metadata={
            "help": "Whether to clear checkpoint directory after training. If set true, will delete the optimizer, "
                    "scheduler, and train state at last, only keep the checkpoint of model."
        },
    )

    # 保存
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Limit the total amount of checkpoints."},
    )


OPTIM_CLS_MAP = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

SCHEDULER_CLS_MAP = {
    "linear": SchedulerType.LINEAR,
    "cosine": SchedulerType.COSINE,
    "cosine_with_restarts": SchedulerType.COSINE_WITH_RESTARTS,
    "polynomial": SchedulerType.POLYNOMIAL,
    "constant": SchedulerType.CONSTANT,
    "constant_with_warmup": SchedulerType.CONSTANT_WITH_WARMUP,
    "inverse_sqrt": SchedulerType.INVERSE_SQRT,
    "reduce_on_plateau": SchedulerType.REDUCE_ON_PLATEAU,
}


def build_optimizer(
    optimizer_name: str, optimizer_specific_kwargs: dict, trainable_params
):
    """按名字构造 :class:`torch.optim.Optimizer`。

    Args:
        optimizer_name: 优化器名，必须是 :data:`OPTIM_CLS_MAP` 的键（大小写不敏感，
            调用方需先 lower）。
        optimizer_specific_kwargs: 透传给优化器构造器的额外关键字参数。
        trainable_params: 可训练参数列表（一般取 ``[p for p in model.parameters() if p.requires_grad]``）。

    Returns:
        构造好的 :class:`torch.optim.Optimizer`。

    Raises:
        AssertionError: ``optimizer_name`` 不在 :data:`OPTIM_CLS_MAP` 时。
    """
    assert optimizer_name in OPTIM_CLS_MAP, f"{optimizer_name} not supported yet."
    return OPTIM_CLS_MAP[optimizer_name](
        params=trainable_params,
        **optimizer_specific_kwargs,
    )


def build_scheduler(
    scheduler_name: str,
    optimizer: Optimizer,
    num_warmup_steps: int = None,
    num_training_steps: int = None,
    scheduler_specific_kwargs=None,
):
    """按名字构造 :func:`transformers.get_scheduler`。

    Args:
        scheduler_name: 调度器名，必须是 :data:`SCHEDULER_CLS_MAP` 的键。
        optimizer: 已构造好的 :class:`torch.optim.Optimizer`。
        num_warmup_steps: 预热步数。
        num_training_steps: 总训练步数。
        scheduler_specific_kwargs: 透传给 :func:`transformers.get_scheduler` 的额外参数。

    Returns:
        构造好的 LR scheduler。

    Raises:
        AssertionError: ``scheduler_name`` 不在 :data:`SCHEDULER_CLS_MAP` 时。
    """
    assert scheduler_name in SCHEDULER_CLS_MAP, f"{scheduler_name} not supported yet."
    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}

    from transformers import get_scheduler  # 局部 import 避免顶层强制依赖

    return get_scheduler(
        name=SCHEDULER_CLS_MAP[scheduler_name],
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs,
    )


class Stage(Enum):
    """训练阶段枚举。

    - ``TRAIN`` —— 训练循环中的 step log
    - ``EVAL`` —— 验证集评估
    - ``TEST`` —— 测试集评估
    """

    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
