# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from bignote_gpt.models.classic.transformer_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(
        self,
        vocabulary_count : int,
        max_sequence_size: int,
        embedding_size   : int,
        block_count      : int,
        head_count       : int,
        dropout          : float):
        super().__init__()
        self.__token_embedding    = nn.Embedding(vocabulary_count, embedding_size)
        self.__position_embedding = nn.Embedding(max_sequence_size, embedding_size)
        self.__dropout            = nn.Dropout(dropout)
        self.__blocks             = nn.ModuleList([TransformerBlock(
            max_sequence_size,
            embedding_size,
            head_count,
            dropout) for _ in range(block_count)])
        self.__final_layer_norm   = nn.LayerNorm(embedding_size, bias=False)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        """
                        Output
                           ^
                           |
                 +-------------------+
            [6]  |     LayerNorm     |
                 +-------------------+
                           ^
                           |
                 +-------------------+
                 | Transformer Block |
                 |      Layer L      |
                 +-------------------+
                           ^
                           |
                 +-------------------+
                 | Transformer Block |
                 |     Layer ...     |
                 +-------------------+
                           ^
                           |
                 +-------------------+
            [5]  | Transformer Block |
                 |      Layer 1      |
                 +-------------------+
                           ^
                           |
                 +-------------------+
            [4]  |      Dropout      |
                 +-------------------+
                           ^
                           |     +---------------------+
            [3]           (+)<---| Positional Encoding |  [2]
                           ^     +---------------------+
                           |
                 +-------------------+
            [1]  |  Input Embedding  |
                 +-------------------+
                           ^
                           |
                         Input
        """
        # input.shape == (batch_size, sequence_size)

        device = input.device

        sequence_size = input.shape[1]

        positions = torch.arange(sequence_size, dtype=torch.long, device=device)

        # positions.shape == (sequence_size,)
        # positions       == [0, 1, 2, ...]

        encoded_tokens   = self.__token_embedding(input)        # [1]
        encoded_position = self.__position_embedding(positions) # [2]

        output = encoded_tokens + encoded_position              # [3]

        # output.shape == (batch_size, sequence_size, embedding_size)

        output = self.__dropout(output)                         # [4]

        for block in self.__blocks:                             # [5]
            output = block(output)

        output = self.__final_layer_norm(output)                # [6]

        return output
