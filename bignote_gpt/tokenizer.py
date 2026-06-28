# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod

class Tokenizer(ABC):
    @property
    @abstractmethod
    def vocabulary_count(self) -> int: ...

    @classmethod
    @abstractmethod
    def create(cls) -> Tokenizer: ...

    @abstractmethod
    def encode(self, text:str) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens:list[int]) -> str: ...
