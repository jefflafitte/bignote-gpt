# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod

from bignote_gpt.model import Model

class Optimizer(ABC):
    @classmethod
    @abstractmethod
    def create(cls, model:Model, learning_rate:float, configDict:dict) -> Optimizer: ...
