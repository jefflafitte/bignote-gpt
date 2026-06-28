# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from pathlib import Path

import torch

from bignote_gpt.checkpoint import load_checkpoint
from bignote_gpt.generator import Generator
from bignote_gpt.seeding import set_seed

def generate(
    checkpoint_path   : Path,
    seed              : int|None,
    deterministic_cuda: bool,
    prompt            : str,
    stop_sequence     : str,
    stop_count        : int,
    max_token_count   : int,
    temperature       : float,
    top_k             : int,
    device_name       : str) -> None:
    cuda_requested = (device_name == "cuda")
    using_cuda     = cuda_requested and torch.cuda.is_available()

    if seed is not None:
        set_seed(seed, using_cuda and deterministic_cuda)

    device = torch.device("cuda" if using_cuda else "cpu")

    tokenizer, model = load_checkpoint(checkpoint_path, device)

    generator = Generator(tokenizer=tokenizer, model=model, device=device)

    print(generator.generate(
        prompt          = prompt,
        stop_sequence   = stop_sequence,
        stop_count      = stop_count,
        max_token_count = max_token_count,
        temperature     = temperature,
        top_k           = top_k))
