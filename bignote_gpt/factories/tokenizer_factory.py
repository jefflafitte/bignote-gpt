# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from bignote_gpt.factory import Factory
from bignote_gpt.tokenizers.byte_level_tokenizer import ByteLevelTokenizer

class _TokenizerFactory(Factory):
    def __init__(self):
        super().__init__({"byte_level": ByteLevelTokenizer})

tokenizer_factory = _TokenizerFactory()
