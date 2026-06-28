# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import torch.nn as nn

class Model(nn.Module, metaclass=ABCMeta):
    @property
    @abstractmethod
    def max_sequence_size(self) -> int: ...

    @classmethod
    @abstractmethod
    def create(cls, vocabulary_count:int, configDict:dict) -> Model: ...
