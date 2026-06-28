# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

import os
import random

import torch

def set_seed(seed: int, deterministic_cuda: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_cuda:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
