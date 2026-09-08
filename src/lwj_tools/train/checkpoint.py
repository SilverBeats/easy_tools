"""Checkpoint 保存 / 加载 / 早停比较。

四个模块级常量 :data:`MODEL` / :data:`OPTIMIZER` / :data:`SCHEDULER` /
:data:`TRAIN_STATE` 是 checkpoint 文件名（不带扩展名）。Trainer 类上的同名
类属性也指向这些字符串（见 :mod:`lwj_tools.train.trainer`）。

Example:
    >>> # 在 Trainer 子类的 train() 流程里调用
    >>> from lwj_tools.train.checkpoint import save_checkpoint, become_better
    >>> eval_result = trainer.evaluate_model(trainer._val_loader)
    >>> if become_better(trainer, eval_result[config.golden_metric]):
    ...     save_checkpoint(trainer, eval_result)
"""
import os

import torch

from ..common.files import rm_dir

# checkpoint 文件名（无扩展名）
MODEL = "model"
OPTIMIZER = "optimizer"
SCHEDULER = "scheduler"
TRAIN_STATE = "train_state"


def save_checkpoint(trainer, eval_result: dict):
    """把当前 model / optimizer / scheduler / train_states 写入 checkpoint 目录。

    同时更新 :attr:`Trainer.train_states` 中的 ``best_golden_metric_value`` 与
    ``best_ckpt_paths``；超出 ``save_total_limit`` 时丢弃最早 checkpoint。

    Args:
        trainer: :class:`Trainer` 实例。
        eval_result: 评估结果字典（含 :attr:`TrainingArguments.golden_metric` 字段）。
    """
    steps = trainer.train_states["steps"]
    output_dir = os.path.join(trainer.config.output_dir, f"checkpoint-{steps}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        trainer.model.state_dict(),
        os.path.join(output_dir, MODEL + ".pt"),
    )
    torch.save(
        trainer._optimizer.state_dict(),
        os.path.join(output_dir, OPTIMIZER + ".pt"),
    )
    torch.save(
        trainer.train_states,
        os.path.join(output_dir, TRAIN_STATE + ".pt"),
    )
    if trainer._scheduler:
        torch.save(
            trainer._scheduler.state_dict(),
            os.path.join(output_dir, SCHEDULER + ".pt"),
        )
    from ..common.logging import get_logger  # 局部 import 避免循环

    logger = get_logger("lwj_tools")
    logger.info(f"Model saved at: {output_dir}")

    trainer.train_states["best_golden_metric_value"] = eval_result[
        trainer.config.golden_metric
    ]
    trainer.train_states["best_ckpt_paths"].append(output_dir)

    if len(trainer.train_states["best_ckpt_paths"]) > trainer.config.save_total_limit:
        rm_dir(trainer.train_states["best_ckpt_paths"].pop(0))


def load_checkpoint(trainer):
    """从 :attr:`TrainingArguments.resume_checkpoint_path` 恢复训练状态。

    同时把 :attr:`TrainingArguments.output_dir` 替换为 checkpoint 所在目录。

    Args:
        trainer: :class:`Trainer` 实例。
    """
    ckpt_dir = trainer.config.resume_checkpoint_path
    trainer.config.output_dir = os.path.dirname(ckpt_dir)

    optimizer_path = os.path.join(ckpt_dir, OPTIMIZER + ".pt")
    scheduler_path = os.path.join(ckpt_dir, SCHEDULER + ".pt")
    train_state_path = os.path.join(ckpt_dir, TRAIN_STATE + ".pt")
    model_ckpt_path = os.path.join(ckpt_dir, MODEL + ".pt")

    trainer._optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
    trainer.model.load_state_dict(torch.load(model_ckpt_path, weights_only=True))
    if trainer.config.warmup_steps > 0:
        trainer._scheduler.load_state_dict(
            torch.load(scheduler_path, weights_only=True),
        )
    trainer.train_states.update(torch.load(train_state_path, weights_only=True))


def become_better(trainer, golden_metric_value: float) -> bool:
    """判断 ``golden_metric_value`` 是否优于历史最优。

    Args:
        trainer: :class:`Trainer` 实例。
        golden_metric_value: 当前评估指标值。

    Returns:
        :data:`True` 表示当前结果更优（按 :attr:`TrainingArguments.lower_is_better`
        决定方向）。
    """
    lower_is_better = trainer.config.lower_is_better

    if lower_is_better:
        return golden_metric_value < trainer.train_states["best_golden_metric_value"]
    else:
        return golden_metric_value > trainer.train_states["best_golden_metric_value"]
