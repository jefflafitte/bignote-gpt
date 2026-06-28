# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from bignote_gpt.models.classic.feed_forward import FeedForward
from bignote_gpt.models.classic.masked_self_attention import MaskedSelfAttention

class TransformerBlock(nn.Module):
    """
                 Output
                    ^
                    |
        [6]        (+)<-------+
                    ^         |
                    |         |
             +-------------+  |
        [5]  | FeedForward |  |
             +-------------+  |
                    ^         |
                    |         |
                    +---------+
                    |
             +-------------+
        [4]  |  LayerNorm  |
             +-------------+
                    ^
                    |
        [3]        (+)<-------+
                    ^         |
                    |         |
             +-------------+  |
        [2]  |  Attention  |  |
             +-------------+  |
                    ^         |
                    |         |
             +-------------+  |
        [1]  |  LayerNorm  |  |
             +-------------+  |
                    ^         |
                    |         |
                    +---------+
                    |
                  Input
    """
    def __init__(
        self,
        max_sequence_size: int,
        embedding_size   : int,
        head_count       : int,
        dropout          : float):
        super().__init__()
        self.__layer_norm1  = nn.LayerNorm(embedding_size, bias=False)
        self.__attention    = MaskedSelfAttention(
            max_sequence_size,
            embedding_size,
            head_count,
            dropout)
        self.__layer_norm2  = nn.LayerNorm(embedding_size, bias=False)
        self.__feed_forward = FeedForward(embedding_size, dropout)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        # input.shape == (batch_size, sequence_size, embedding_size)
        layer_output = self.__layer_norm1(input)         # [1]
        layer_output = self.__attention(layer_output)    # [2]
        output       = input + layer_output              # [3]
        layer_output = self.__layer_norm2(output)        # [4]
        layer_output = self.__feed_forward(layer_output) # [5]
        output       = output + layer_output             # [6]
        # output.shape == (batch_size, sequence_size, embedding_size)
        return output
