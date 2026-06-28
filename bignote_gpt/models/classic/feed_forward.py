# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embedding_size:int, dropout:float):
        super().__init__()
        self.__fully_connected = nn.Linear(embedding_size, 4*embedding_size, bias=False)
        self.__relu            = nn.ReLU()
        self.__projection      = nn.Linear(4*embedding_size, embedding_size, bias=False)
        self.__dropout         = nn.Dropout(dropout)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        """
                     Output
                        ^
                        |
                   +---------+
            [4]    | Dropout |
                   +---------+
                        ^
                        |
                   +---------+
            [3]   |  Linear   |
                 +-------------+
                        ^
                        |
                 +-------------+
            [2]  |    Relu     |
                 +-------------+
                        ^
                        |
                 +-------------+
            [1]   |  Linear   |
                   +---------+
                        ^
                        |
                      Input
        """
        # input.shape == (batch_size, sequence_size, embedding_size)
        output = self.__fully_connected(input) # [1]
        # output.shape == (batch_size, sequence_size, 4*embedding_size)
        output = self.__relu(output)           # [2]
        output = self.__projection(output)     # [3]
        # output.shape == (batch_size, sequence_size, embedding_size)
        output = self.__dropout(output)        # [4]
        return output
