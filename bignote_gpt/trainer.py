# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

import math

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from bignote_gpt.dataset import Dataset
from bignote_gpt.defaults import Defaults
from bignote_gpt.model import Model
from bignote_gpt.training_reporter import TrainingReporter

@dataclass
class TrainerConfig:
    batch_size       : int   = field(default=Defaults.batch_size       , metadata={"help": Defaults.help.batch_size       })
    iteration_limit  : int   = field(default=Defaults.iteration_limit  , metadata={"help": Defaults.help.iteration_limit  })
    warmup_count     : int   = field(default=Defaults.warmup_count     , metadata={"help": Defaults.help.warmup_count     })
    learning_rate    : float = field(default=Defaults.learning_rate    , metadata={"help": Defaults.help.learning_rate    })
    min_learning_rate: float = field(default=Defaults.min_learning_rate, metadata={"help": Defaults.help.min_learning_rate})
    evaluation_period: int   = field(default=Defaults.evaluation_period, metadata={"help": Defaults.help.evaluation_period})
    evaluation_count : int   = field(default=Defaults.evaluation_count , metadata={"help": Defaults.help.evaluation_count })
    patience         : int   = field(default=Defaults.patience         , metadata={"help": Defaults.help.patience         })
    gradient_clip    : float = field(default=Defaults.gradient_clip    , metadata={"help": Defaults.help.gradient_clip    })

@dataclass
class TrainingResult:
    trained_iteration:int
    training_loss    : float
    validation_loss  : float

class Trainer:
    def __init__(
        self,
        config          : TrainerConfig,
        model           : Model,
        optimizer       : torch.optim.Optimizer,
        training_data   : Dataset,
        validation_data : Dataset,
        device          : torch.device,
        reporter        : TrainingReporter):
        self.__config          = config
        self.__model           = model
        self.__optimizer       = optimizer
        self.__training_data   = training_data
        self.__validation_data = validation_data
        self.__device          = device
        self.__reporter        = reporter
        self.__use_amp         = (device.type == "cuda")
        self.__amp_dtype       = torch.bfloat16 if (self.__use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        self.__scaler          = torch.amp.GradScaler(
            device.type,
            enabled=(self.__use_amp and (self.__amp_dtype == torch.float16)))

    def train(self) -> TrainingResult:
        self.__reporter.on_start_training()

        min_training_loss   = float("inf")
        min_validation_loss = float("inf")
        best_state          = None
        patience_count      = 0
        trained_iteration   = 0

        for iteration in range(self.__config.iteration_limit + 1):
            learning_rate = self.__calculate_learning_rate(iteration)

            for param_group in self.__optimizer.param_groups:
                param_group["lr"] = learning_rate

            if iteration % self.__config.evaluation_period == 0:
                training_loss   = self.estimate_loss(self.__training_data)
                validation_loss = self.estimate_loss(self.__validation_data)
                is_best         = validation_loss < min_validation_loss

                self.__reporter.on_evaluation(iteration, learning_rate, training_loss, validation_loss, is_best)

                if is_best:
                    min_training_loss       = training_loss
                    min_validation_loss     = validation_loss
                    best_state              = {name: tensor.detach().cpu().clone()
                        for name, tensor in self.__model.state_dict().items()}
                    patience_count          = 0
                    trained_iteration       = iteration
                else:
                    patience_count += 1
                    if (self.__config.patience > 0) and (patience_count >= self.__config.patience):
                        self.__reporter.on_lost_patience()
                        break

            input, targets = self.__training_data.get_batch(
                self.__config.batch_size,
                self.__model.max_sequence_size,
                self.__device)

            # input.shape   == (batch_size, max_sequence_size)
            # targets.shape == (batch_size, max_sequence_size)

            with torch.autocast(device_type=self.__device.type, dtype=self.__amp_dtype, enabled=self.__use_amp):
                _, loss = self.__model(input, targets)

            self.__optimizer.zero_grad(set_to_none=True)
            self.__scaler.scale(loss).backward()
            self.__scaler.unscale_(self.__optimizer)
            nn.utils.clip_grad_norm_(self.__model.parameters(), self.__config.gradient_clip)
            self.__scaler.step(self.__optimizer)
            self.__scaler.update()

        if best_state is not None:
            self.__model.load_state_dict(best_state)

        self.__reporter.on_finish_training(iteration, trained_iteration, min_validation_loss)

        return TrainingResult(
            trained_iteration = trained_iteration,
            training_loss     = min_training_loss,
            validation_loss   = min_validation_loss)

    @torch.no_grad()
    def estimate_loss(self, data:Dataset) -> float:
        self.__model.eval()
        losses = torch.zeros(self.__config.evaluation_count)
        for i in range(self.__config.evaluation_count):
            input, targets = data.get_batch(self.__config.batch_size, self.__model.max_sequence_size, self.__device)
            with torch.autocast(device_type=self.__device.type, dtype=self.__amp_dtype, enabled=self.__use_amp):
                _, loss = self.__model(input, targets)
            losses[i] = loss.item()
        self.__model.train()
        return losses.mean().item()

    def __calculate_learning_rate(self, iteration:int) -> float:
        if iteration < self.__config.warmup_count:
            return self.__config.learning_rate*(iteration + 1)/self.__config.warmup_count
        progress = (iteration - self.__config.warmup_count)/max(1, self.__config.iteration_limit - self.__config.warmup_count)
        coefficient = 0.5*(1.0 + math.cos(math.pi * progress))
        return self.__config.min_learning_rate + coefficient*(self.__config.learning_rate - self.__config.min_learning_rate)
