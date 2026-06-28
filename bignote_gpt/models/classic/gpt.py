# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torch.nn import functional as F

from bignote_gpt.model import Model

from bignote_gpt.models.classic.transformer import Transformer

@dataclass
class GPTConfig:
    max_sequence_size: int   = field(default=256, metadata={"help": "Maximum context length in tokens."                                       })
    embedding_size   : int   = field(default=384, metadata={"help": "Embedding size."                                                         })
    block_count      : int   = field(default=6  , metadata={"help": "Number of transformer blocks."                                           })
    head_count       : int   = field(default=6  , metadata={"help": "Number of attention heads (must be an integer factor of embedding_size)."})
    dropout          : float = field(default=0.2, metadata={"help": "Dropout probability."                                                    })

class GPT(Model):
    def __init__(self, vocabulary_count:int, config:GPTConfig):
        super().__init__()
        self.__config      = config
        self.__transformer = Transformer(
            vocabulary_count  = vocabulary_count,
            max_sequence_size = config.max_sequence_size,
            embedding_size    = config.embedding_size,
            block_count       = config.block_count,
            head_count        = config.head_count,
            dropout           = config.dropout)
        self.__head        = nn.Linear(config.embedding_size, vocabulary_count, bias=False)

    @property
    def max_sequence_size(self) -> int:
        return self.__config.max_sequence_size

    @classmethod
    def create(cls, vocabulary_count:int, configDict:dict) -> GPT:
        return GPT(vocabulary_count, GPTConfig(**configDict))

    def forward(self, input:torch.Tensor, targets:torch.Tensor|None=None) -> tuple[torch.Tensor, torch.Tensor|None]:
        """
                     Output
                        ^
                        |
                 +-------------+
            [2]  |   Linear    |
                 +-------------+
                        ^
                        |
                 +-------------+
            [1]  | Transformer |
                 +-------------+
                        ^
                        |
                      Input
        """
        # input.shape == (batch_size, sequence_size)

        output = self.__transformer(input)           # [1]

        # output.shape == (batch_size, sequence_size, embedding_size)

        if targets is not None:
            logits = self.__head(output)             # [2]
            loss   = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1)
        else:
            logits = self.__head(output[:, [-1], :]) # [2]
            loss   = None

        return logits, loss
