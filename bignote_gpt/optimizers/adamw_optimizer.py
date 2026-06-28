# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from bignote_gpt.model import Model
from bignote_gpt.optimizer import Optimizer

@dataclass
class AdamWOptimizerConfig:
    weight_decay: float               = field(default=0.1,         metadata={"help": "Weight decay (applied to 2-D weights only)."})
    betas       : tuple[float, float] = field(default=(0.9, 0.99), metadata={"help": "AdamW beta coefficients."                   })
    eps         : float               = field(default=1e-8,        metadata={"help": "AdamW epsilon for numerical stability."     })
    def __post_init__(self):
        self.betas = tuple(self.betas)

class AdamWOptimizer(torch.optim.AdamW, Optimizer):
    @classmethod
    def create(cls, model:Model, learning_rate:float, configDict:dict) -> AdamWOptimizer:
        config = AdamWOptimizerConfig(**configDict)

        decay_parameters    = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
        no_decay_parameters = [parameter for parameter in model.parameters() if parameter.dim() <  2]

        return cls(
            [
                {"params": decay_parameters,    "weight_decay": config.weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ],
            lr    = learning_rate,
            betas = config.betas,
            eps   = config.eps)
