# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

import torch

from bignote_gpt.tokenizer import Tokenizer

class Dataset:
    def __init__(self, text:str, tokenizer:Tokenizer):
        self.__data             = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.__count            = len(self.__data)
        self.__vocabulary_count = tokenizer.vocabulary_count

    @property
    def vocabulary_count(self) -> int:
        return self.__vocabulary_count

    @property
    def data(self) -> torch.Tensor:
        return self.__data

    def get_batch(self, batch_size:int, sequence_size:int, device:torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.randint(self.__count - sequence_size, (batch_size,))
        input   = torch.stack([self.__data[offset:offset + sequence_size] for offset in offsets]).to(device)
        targets = torch.stack([self.__data[offset + 1:offset + sequence_size+1] for offset in offsets]).to(device)
        return input, targets
