"""训练 / 评估 / 测试主循环与优化器钩子。

所有函数签名形如 ``run_xxx(trainer)`` 或 ``xxx(trainer, batch)``，把 trainer
实例作为显式第一参数传入。Trainer 类上有同名方法转发到这些函数（见
:mod:`lwj_tools.train.trainer`）。

公共函数：

- :func:`model_forward` —— 把 batch 搬到 device 后调模型
- :func:`before_optim_lr_scheduler` / :func:`optim_lr_scheduler` / :func:`after_optim_lr_scheduler` ——
  优化器 step 前后钩子（最后一个默认 no-op）
- :func:`make_infinite_train_loader` —— 无限迭代 train loader
- :func:`evaluate_model_on_test` —— 在 test loader 上评估所有 best checkpoint
- :func:`run_train_loop` —— 主训练循环
"""
import os

import torch
from tqdm import tqdm

from ..common.files import rm_file
from ..common.logging import get_logger
from ..nn.helpers import data_2_device
from .checkpoint import become_better, save_checkpoint
from .optimizer import Stage

LOGGER = get_logger("lwj_tools")


def model_forward(trainer, batch) -> dict:
    """把 batch 搬到 device 后调用 model，返回 forward 结果。

    Args:
        trainer: :class:`Trainer` 实例。
        batch: 训练数据 batch。

    Returns:
        模型 forward 输出（一般是 :class:`dict`）。
    """
    device_batch = data_2_device(batch, trainer.config.device)
    if isinstance(device_batch, dict):
        forward_dict = trainer.model(**device_batch)
    else:
        forward_dict = trainer.model(device_batch)
    return forward_dict


def before_optim_lr_scheduler(trainer):
    """优化器 step 之前的钩子（默认实现：梯度裁剪）。

    Args:
        trainer: :class:`Trainer` 实例。
    """
    if trainer.config.max_grad_norm >= 0:
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(),
            max_norm=trainer.config.max_grad_norm,
        )


def optim_lr_scheduler(trainer):
    """执行 optimizer.step + zero_grad，以及可选的 scheduler.step。

    Args:
        trainer: :class:`Trainer` 实例。
    """
    trainer._optimizer.step()
    trainer._optimizer.zero_grad()
    if trainer.config.warmup_steps > 0:
        trainer._scheduler.step()


def after_optim_lr_scheduler(trainer):
    """优化器 step 之后的钩子（默认无操作，供子类覆写）。

    Args:
        trainer: :class:`Trainer` 实例。
    """
    pass


def make_infinite_train_loader(trainer):
    """把 train loader 包装成无限迭代器（用于训练循环）。

    Args:
        trainer: :class:`Trainer` 实例。

    Yields:
        batch。
    """
    while True:
        for batch in trainer._train_loader:
            yield batch


def evaluate_model_on_test(trainer):
    """对 :attr:`Trainer.train_states` 里所有 ``best_ckpt_paths`` 在 test loader 上评估。

    评估结果写入各 checkpoint 目录下的 ``eval_on_test.jsonl``。

    Args:
        trainer: :class:`Trainer` 实例。
    """
    for ckpt_dir in trainer.train_states["best_ckpt_paths"]:
        LOGGER.info(
            f"Evaluate {os.path.join(ckpt_dir, 'model.pt')} on the test set",
        )
        trainer.model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "model.pt")),
        )
        eval_result = trainer.evaluate_model(trainer._test_loader)
        fp = open(
            os.path.join(ckpt_dir, "eval_on_test.jsonl"),
            "w",
            encoding="utf-8",
            buffering=1,
        )
        trainer._log(eval_result, Stage.TEST, fp)
        fp.flush()
        fp.close()


def run_train_loop(trainer):
    """主训练循环：step / eval / save / early-stop 直到达到 ``total_steps``。

    Args:
        trainer: :class:`Trainer` 实例。
    """
    pbar = (
        tqdm(
            total=trainer.config.total_steps,
            desc="Training",
            dynamic_ncols=True,
            leave=False,
        )
        if trainer.config.use_pbar
        else None
    )

    train_loader = make_infinite_train_loader(trainer)
    if trainer.train_states["global_steps"] != 0:
        for _ in range(trainer.train_states["global_steps"]):
            next(train_loader)

    if pbar:
        pbar.update(trainer.train_states["global_steps"])
        pbar.refresh()

    while True:
        forward_dict = model_forward(trainer, next(train_loader))
        loss = (
                forward_dict[trainer.config.loss_field]
                / trainer.config.gradient_accumulation_steps
        )
        loss.backward()

        trainer.train_states["accumulate_loss"] += loss.item()
        trainer.train_states["global_steps"] += 1

        if (
                trainer.train_states["global_steps"]
                % trainer.config.gradient_accumulation_steps
                == 0
        ):
            before_optim_lr_scheduler(trainer)
            optim_lr_scheduler(trainer)
            after_optim_lr_scheduler(trainer)
            trainer.train_states["steps"] += 1

            if trainer.config.verbose:
                trainer._log(forward_dict, Stage.TRAIN)

            trainer.train_states["accumulate_loss"] = 0

            if pbar:
                pbar.update(1)
                pbar.refresh()

        if trainer.train_states["steps"] % trainer.config.cache_empty_steps == 0:
            torch.cuda.empty_cache()

        if trainer.train_states["steps"] % trainer.config.eval_steps == 0:
            eval_result = trainer.evaluate_model(trainer._val_loader)
            trainer._log(eval_result, Stage.EVAL)
            if not become_better(trainer, eval_result[trainer.config.golden_metric]):
                if trainer.config.patience > 0:
                    trainer.train_states["patience"] -= 1
            else:
                save_checkpoint(trainer, eval_result)
                if trainer.config.patience > 0:
                    trainer.train_states["patience"] = trainer.config.patience

        if trainer.config.patience > 0 and trainer.train_states["patience"] == 0:
            break

        if trainer.train_states["steps"] >= trainer.config.total_steps:
            break

    if pbar:
        pbar.close()

    # 收尾：若最后一轮没 eval（off-by-one），补一次 eval
    if (
            trainer.config.patience == 0
            and trainer.train_states["steps"] % trainer.config.eval_steps != 0
    ) or (trainer.config.patience > 0 and trainer.train_states["patience"] != 0):
        eval_result = trainer.evaluate_model(trainer._val_loader)
        trainer._log(eval_result, Stage.EVAL)
        if become_better(trainer, eval_result[trainer.config.golden_metric]):
            save_checkpoint(trainer, eval_result)

    if trainer.config.eval_best_model_on_test:
        evaluate_model_on_test(trainer)

    if trainer.config.clear_ckpt_dir:
        for ckpt_path in trainer.train_states["best_ckpt_paths"]:
            rm_file(os.path.join(ckpt_path, "optimizer.pt"))
            rm_file(os.path.join(ckpt_path, "scheduler.pt"))
            rm_file(os.path.join(ckpt_path, "train_state.pt"))

    if trainer.config.verbose:
        trainer._train_logger.close()
        if trainer._val_loader is not None:
            trainer._eval_logger.close()
        if trainer.config.adopt_tensorboard:
            trainer._tb_writer.close()
