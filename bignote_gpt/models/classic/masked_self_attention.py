# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

class MaskedSelfAttention(nn.Module):
    mask: torch.Tensor

    def __init__(self, max_sequence_size:int, embedding_size:int, head_count:int, dropout:float):
        super().__init__()

        assert embedding_size%head_count == 0

        self.__head_count           = head_count
        self.__head_dimension_count = embedding_size//head_count

        self.__query_projection = nn.Linear(embedding_size, embedding_size, bias=False)
        self.__key_projection   = nn.Linear(embedding_size, embedding_size, bias=False)
        self.__value_projection = nn.Linear(embedding_size, embedding_size, bias=False)

        mask = torch.ones(max_sequence_size, max_sequence_size)
        mask = torch.tril(mask)
        mask = mask.view(1, 1, max_sequence_size, max_sequence_size)

        self.register_buffer("mask", mask)

        self.__attention_dropout = nn.Dropout(dropout)

        self.__output_projection = nn.Linear(embedding_size, embedding_size, bias=False)

        self.__output_dropout = nn.Dropout(dropout)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        """
                                     Output
                                        ^
                                        |
                                   +---------+
            [8]                    | Dropout |
                                   +---------+
                                        ^
                                        |
                                   +---------+
            [7]                    | Linear  |
                                   +---------+
                                        ^
                                        |
                        +---------------+---------------+
                        |               |               |
                 +-------------+ +-------------+ +-------------+
            [6]  |   Matmul    | |   Matmul    | |   Matmul    |
                 +-------------+ +-------------+ +-------------+
                      ^      ^        ^      ^        ^      ^
                      |      |        |      |        |      |
                 +---------+ |   +---------+ |   +---------+ |
            [5]  | Dropout | |   | Dropout | |   | Dropout | |
                 +---------+ |   +---------+ |   +---------+ |
                      ^      |        ^      |        ^      |
                      |      |        |      |        |      |
                 +---------+ |   +---------+ |   +---------+ |
            [4]  | Softmax | |   | Softmax | |   | Softmax | |
                 +---------+ |   +---------+ |   +---------+ |
                      ^      |        ^      |        ^      |
                      |      |        |      |        |      |
                 +---------+ |   +---------+ |   +---------+ |
            [3]  |  Mask   | |   |  Mask   | |   |  Mask   | |
                 +---------+ |   +---------+ |   +---------+ |
                      ^      |        ^      |        ^      |
                      |      |        |      |        |      |
                 +---------+ |   +---------+ |   +---------+ |
            [2]  | Matmul  | |   | Matmul  | |   | Matmul  | |
                 +---------+ |   +---------+ |   +---------+ |
                     ^ ^     |       ^ ^     |       ^ ^     |
                     | |     |       | |     |       | |     |
                 +-----------------------------------------------+
            [1]   |  Q K     V       Q K     V       Q K     V  |
                   |  Head 1         Head ...         Head H   |
                    |                                         |
                     |               Linear                  |
                      +-------------------------------------+
                                        ^
                                        |
                                      Input
        """
        # input.shape == (batch_size, sequence_size, embedding_size)

        batch_size, sequence_size, _ = input.shape

        query = self.__query_projection(input) # [1]
        key   = self.__key_projection  (input) # [1]
        value = self.__value_projection(input) # [1]

        # query.shape == (batch_size, sequence_size, embedding_size)
        # key  .shape == (batch_size, sequence_size, embedding_size)
        # value.shape == (batch_size, sequence_size, embedding_size)

        query = query.view(batch_size, sequence_size, self.__head_count, self.__head_dimension_count).transpose(1, 2)
        key   = key  .view(batch_size, sequence_size, self.__head_count, self.__head_dimension_count).transpose(1, 2)
        value = value.view(batch_size, sequence_size, self.__head_count, self.__head_dimension_count).transpose(1, 2)

        # query.shape == (batch_size, head_count, sequence_size, head_dimension_count)
        # key  .shape == (batch_size, head_count, sequence_size, head_dimension_count)
        # value.shape == (batch_size, head_count, sequence_size, head_dimension_count)

        # (batch_size, head_count, sequence_size, head_dimension_count) X
        #     (batch_size, head_count, head_dimension_count, sequence_size) ->
        #     (batch_size, head_count, sequence_size, sequence_size)
        attention = (query@key.transpose(-2, -1))/(self.__head_dimension_count**0.5) # [2]

        # attention.shape == (batch_size, head_count, sequence_size, sequence_size)

        attention = attention.masked_fill(
            self.mask[:, :, :sequence_size, :sequence_size] == 0, float('-inf'))     # [3]
        attention = torch.softmax(attention, dim=-1)                                 # [4]
        attention = self.__attention_dropout(attention)                              # [5]

        # (batch_size, head_count, sequence_size, sequence_size) X
        #     (batch_size, head_count, sequence_size, head_dimension_count) ->
        #     (batch_size, head_count, sequence_size, head_dimension_count)
        output = attention@value                                                     # [6]

        # output.shape == (batch_size, head_count, sequence_size, head_dimension_count)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_size, -1)

        # output.shape == (batch_size, sequence_size, embedding_size)

        output = self.__output_projection(output)                                    # [7]
        output = self.__output_dropout(output)                                       # [8]

        return output
