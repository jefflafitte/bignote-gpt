# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from bignote_gpt.factory import Factory
from bignote_gpt.models.classic.gpt import GPT

class _ModelFactory(Factory):
    def __init__(self):
        super().__init__({"classic_gpt": GPT})

model_factory = _ModelFactory()
