#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from transformers import SchedulerType, get_scheduler, set_seed

from .constant import LOGGER
from .io import FileWriter
from .model import convert_data_to_normal_type, data_2_device
from .tools import clean_dir, rm_dir, rm_file


@dataclass
class TrainingArguments:
    seed: int = field(default=42, metadata={"help": "Random seed."})
    output_dir: str = field(default="output", metadata={"help": "Output directory."})
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Whether to overwrite the output directory."},
    )

    # related to log
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

    # related to dataloader
    train_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for training."},
    )
    eval_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for evaluation."},
    )

    # related to train
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

    # related to save
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
    assert scheduler_name in SCHEDULER_CLS_MAP, f"{scheduler_name} not supported yet."
    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}

    return get_scheduler(
        name=SCHEDULER_CLS_MAP[scheduler_name],
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs,
    )


class Stage(Enum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class Trainer(ABC):
    """
    1) You can override `build_*_loader` to customize how you build the dataloader
    2) val_file_path is not necessary,
    2-1) If you don't set it, the model will be trained until the end
    2-2) If you set the val_file_path and rewrite the `evaluate_model`,
        then you can judge the quality of the model according to the gold indicator you set
    """

    MODEL = "model"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    TRAIN_STATE = "train_state"

    def __init__(
        self,
        model: nn.Module,
        train_file_path: str,
        config: TrainingArguments = None,
        val_file_path: str = None,
        test_file_path: str = None,
    ):
        if config is None:
            config = TrainingArguments()

        self._config = config
        assert self._config.seed >= 0, "seed must be greater than or equal to 0."
        set_seed(self._config.seed)

        if self._config.overwrite_output_dir:
            clean_dir(self._config.output_dir)

        # prepare the dataloaders
        self._train_loader = self.build_train_loader(train_file_path)
        self._val_loader = (
            self.build_val_loader(val_file_path) if val_file_path else None
        )
        self._test_loader = (
            self.build_test_loader(test_file_path) if test_file_path else None
        )

        self._check_and_set_default_config()

        # prepare optimizer and scheduler
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

        # prepare the train states
        # global_steps = steps * gradient_accumulation_steps
        self._train_states = {
            "global_steps": 0,  # record iteration times
            "steps": 0,  # record optimizer step times
            "accumulate_loss": 0,
            "patience": self._config.patience,
            "best_ckpt_paths": [],
            "best_golden_metric_value": (
                float("inf") if self._config.lower_is_better else float("-inf")
            ),
        }

        if self._config.resume_checkpoint_path:
            self._load_checkpoint()

        # prepare log
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

        # save config
        FileWriter.dump(
            asdict(self._config),
            os.path.join(self._config.output_dir, "config.yaml"),
        )

    @property
    def model(self):
        return self._model

    @property
    def config(self):
        return self._config

    @property
    def train_states(self):
        return self._train_states

    def _check_and_set_default_config(self):
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
    def build_train_loader(self, train_file_path: str):
        raise NotImplementedError

    @abstractmethod
    def build_val_loader(self, val_file_path: str):
        raise NotImplementedError

    def build_test_loader(self, test_file_path: str):
        return self.build_val_loader(test_file_path)

    def get_model_trainable_params(self):
        return self._model.parameters()

    def _load_checkpoint(self):
        """resume training"""
        ckpt_dir = self._config.resume_checkpoint_path
        self._config.output_dir = os.path.dirname(ckpt_dir)

        optimizer_path = os.path.join(ckpt_dir, self.OPTIMIZER + ".pt")
        scheduler_path = os.path.join(ckpt_dir, self.SCHEDULER + ".pt")
        train_state_path = os.path.join(ckpt_dir, self.TRAIN_STATE + ".pt")
        model_ckpt_path = os.path.join(ckpt_dir, self.MODEL + ".pt")

        self._optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
        self._model.load_state_dict(torch.load(model_ckpt_path, weights_only=True))
        if self._config.warmup_steps > 0:
            self._scheduler.load_state_dict(
                torch.load(scheduler_path, weights_only=True),
            )
        self._train_states.update(torch.load(train_state_path, weights_only=True))

    def _log(self, result_dict: dict, stage: Stage, custom_logger=None):
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

        # if stage == Stage.TRAIN:
        #     LOGGER.info(f"{stage} steps: {steps}, {log_save_dict}")

        if fp:
            fp.write(json.dumps(log_save_dict, ensure_ascii=False) + "\n")
            fp.flush()

    def _become_better(self, golden_metric_value: float):
        lower_is_better = self._config.lower_is_better

        if lower_is_better:
            return golden_metric_value < self._train_states["best_golden_metric_value"]
        else:
            return golden_metric_value > self._train_states["best_golden_metric_value"]

    def _save_checkpoint(self, eval_result: dict):
        steps = self._train_states["steps"]
        output_dir = os.path.join(self._config.output_dir, f"checkpoint-{steps}")
        os.makedirs(output_dir, exist_ok=True)
        # save
        torch.save(
            self._model.state_dict(),
            os.path.join(output_dir, self.MODEL + ".pt"),
        )
        torch.save(
            self._optimizer.state_dict(),
            os.path.join(output_dir, self.OPTIMIZER + ".pt"),
        )
        torch.save(
            self._train_states,
            os.path.join(output_dir, self.TRAIN_STATE + ".pt"),
        )
        if self._scheduler:
            torch.save(
                self._scheduler.state_dict(),
                os.path.join(output_dir, self.SCHEDULER + ".pt"),
            )
        LOGGER.info(f"Model saved at: {output_dir}")

        # update info
        self._train_states["best_golden_metric_value"] = eval_result[
            self._config.golden_metric
        ]
        self._train_states["best_ckpt_paths"].append(output_dir)

        if len(self._train_states["best_ckpt_paths"]) > self._config.save_total_limit:
            rm_dir(self._train_states["best_ckpt_paths"].pop(0))

    @abstractmethod
    def evaluate_model(self):
        raise NotImplementedError

    def evaluate_model_on_test(self):
        for ckpt_dir in self._train_states["best_ckpt_paths"]:
            LOGGER.info(
                f"Evaluate {os.path.join(ckpt_dir, self.MODEL + 'pt')} on the test set",
            )
            self._model.load_state_dict(
                torch.load(os.path.join(ckpt_dir, self.MODEL + ".pt")),
            )
            eval_result = self.evaluate_model()
            fp = open(
                os.path.join(ckpt_dir, "eval_on_test.jsonl"),
                "w",
                encoding="utf-8",
                buffering=1,
            )
            self._log(eval_result, Stage.TEST, fp)
            fp.flush()
            fp.close()

    def _make_infinite_train_loader(self):
        while True:
            for batch in self._train_loader:
                yield batch

    def model_forward(self, batch) -> dict:
        device_batch = data_2_device(batch, self._config.device)
        if isinstance(device_batch, dict):
            forward_dict = self._model(**device_batch)
        else:
            forward_dict = self._model(device_batch)
        return forward_dict

    def before_optim_lr_scheduler(self):
        if self._config.max_grad_norm >= 0:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                max_norm=self._config.max_grad_norm,
            )

    def optim_lr_scheduler(self):
        self._optimizer.step()
        self._optimizer.zero_grad()
        if self._config.warmup_steps > 0:
            self._scheduler.step()

    def after_optim_lr_scheduler(self):
        pass

    def train(self):
        pbar = (
            tqdm(
                total=self._config.total_steps,
                desc="Training",
                dynamic_ncols=True,
                leave=False,
            )
            if self._config.use_pbar
            else None
        )

        # init training
        train_loader = self._make_infinite_train_loader()
        if self._train_states["global_steps"] != 0:
            for _ in range(self._train_states["global_steps"]):
                next(train_loader)

        if pbar:
            pbar.update(self._train_states["global_steps"])
            pbar.refresh()

        while True:
            forward_dict = self.model_forward(next(train_loader))
            loss = (
                forward_dict[self._config.loss_field]
                / self._config.gradient_accumulation_steps
            )
            loss.backward()

            self._train_states["accumulate_loss"] += loss.item()
            self._train_states["global_steps"] += 1

            if (
                self._train_states["global_steps"]
                % self._config.gradient_accumulation_steps
                == 0
            ):
                self.before_optim_lr_scheduler()
                self.optim_lr_scheduler()
                self.after_optim_lr_scheduler()
                self._train_states["steps"] += 1

                if self._config.verbose:
                    self._log(forward_dict, Stage.TRAIN)

                self._train_states["accumulate_loss"] = 0

                if pbar:
                    pbar.update(1)
                    pbar.refresh()

            if self._train_states["steps"] % self._config.cache_empty_steps == 0:
                torch.cuda.empty_cache()

            if self._train_states["steps"] % self._config.eval_steps == 0:
                eval_result = self.evaluate_model()
                self._log(eval_result, Stage.EVAL)
                if not self._become_better(eval_result[self._config.golden_metric]):
                    if self._config.patience > 0:
                        self._train_states["patience"] -= 1
                else:
                    self._save_checkpoint(eval_result)
                    if self._config.patience > 0:
                        self._train_states["patience"] = self._config.patience

            if self._config.patience > 0 and self._train_states["patience"] == 0:
                break

            if self._train_states["steps"] >= self._config.total_steps:
                break

        if pbar:
            pbar.close()

        # 1) not use early stopping, steps % eval_steps != 0
        # 2) use early stopping, but not break by early stopping
        if (
            self._config.patience == 0
            and self._train_states["steps"] % self._config.eval_steps != 0
        ) or (self._config.patience > 0 and self._train_states["patience"] != 0):
            eval_result = self.evaluate_model()
            self._log(eval_result, Stage.EVAL)
            if self._become_better(eval_result[self._config.golden_metric]):
                self._save_checkpoint(eval_result)

        if self._config.eval_best_model_on_test:
            self.evaluate_model_on_test()

        if self._config.clear_ckpt_dir:
            for ckpt_path in self._train_states["best_ckpt_paths"]:
                rm_file(os.path.join(ckpt_path, self.OPTIMIZER + ".pt"))
                rm_file(os.path.join(ckpt_path, self.SCHEDULER + ".pt"))
                rm_file(os.path.join(ckpt_path, self.TRAIN_STATE + ".pt"))

        if self._config.verbose:
            self._train_logger.close()
            if self._val_loader is not None:
                self._eval_logger.close()
            if self._config.adopt_tensorboard:
                self._tb_writer.close()
