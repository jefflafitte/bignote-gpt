# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from bignote_gpt.factory import Factory
from bignote_gpt.optimizers.adamw_optimizer import AdamWOptimizer

class _OptimizerFactory(Factory):
    def __init__(self):
        super().__init__({"adamw": AdamWOptimizer})

optimizer_factory = _OptimizerFactory()
