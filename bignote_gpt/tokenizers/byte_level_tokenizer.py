# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from __future__ import annotations

from bignote_gpt import tokenizer

class ByteLevelTokenizer(tokenizer.Tokenizer):
    @property
    def vocabulary_count(self) -> int:
        return 256

    @classmethod
    def create(cls) -> ByteLevelTokenizer:
        return ByteLevelTokenizer()

    def encode(self, text:str) -> list[int]:
        return list(text.encode())

    def decode(self, tokens:list[int]) -> str:
        return bytes(tokens).decode(errors="replace")
