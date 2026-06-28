# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bignote_gpt.checkpoint import load_checkpoint
from bignote_gpt.generator import Generator
from bignote_gpt.seeding import set_seed

def interact(
    checkpoint_path   : Path,
    seed              : int|None,
    deterministic_cuda: bool,
    stop_sequence     : str,
    stop_count        : int,
    max_token_count   : int,
    temperature       : float,
    top_k             : int|None,
    device_name       : str) -> None:
    cuda_requested = (device_name == "cuda")
    using_cuda     = cuda_requested and torch.cuda.is_available()

    if seed is not None:
        set_seed(seed, using_cuda and deterministic_cuda)

    device = torch.device("cuda" if using_cuda else "cpu")

    tokenizer, model = load_checkpoint(checkpoint_path, device)
    
    generator = Generator(tokenizer=tokenizer, model=model, device=device)

    settings: dict[str, Any] = {
        "stop-at"    : stop_sequence,
        "stop-count" : stop_count,
        "token-count": max_token_count,
        "temp"       : temperature,
        "top-k"      : top_k}

    print(f"Loaded {checkpoint_path} on {device}.  Enter a prompt, or :help for commands.")

    while True:
        try:
            line = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()

        if not line:
            continue

        if line.startswith(":"):
            if _handle_command(line, settings):
                break
            continue

        print(generator.generate(
            prompt          = line,
            stop_sequence   = settings["stop-at"],
            stop_count      = settings["stop-count"],
            max_token_count = settings["token-count"],
            temperature     = settings["temp"],
            top_k           = settings["top-k"]))

def _handle_command(input:str, settings:dict) -> bool:
    parts    = input.split()
    command  = parts[0]
    argument = parts[1] if len(parts) > 1 else None

    if command == ":stop-at":
        _set_string(settings, "stop-at", argument)
    elif command == ":stop-count":
        _set_int(settings, "stop-count", argument)
    elif command == ":token-count":
        _set_int(settings, "token-count", argument)
    elif command == ":temp":
        _set_float(settings, "temp", argument)
    elif command == ":top-k":
        if argument == "off":
            settings["top-k"] = None
            print("top-k = off")
        else:
            _set_int(settings, "top-k", argument)
    elif command == ":show":
        _print_settings(settings)
    elif command in (":help", ":h"):
        _print_help()
    elif command in (":quit", ":q", ":exit"):
        return True
    else:
        print(f"Unknown command: {command}  (try :help)")

    return False

def _set_int(settings:dict, key:str, argument:str|None) -> None:
    if argument is None:
        print(f"{key} = {settings[key]}")
        return
    try:
        settings[key] = int(argument)
        print(f"{key} = {settings[key]}")
    except ValueError:
        print(f"Expected an integer, got: {argument}")

def _set_float(settings:dict, key:str, argument:str|None) -> None:
    if argument is None:
        print(f"{key} = {settings[key]}")
        return
    try:
        settings[key] = float(argument)
        print(f"{key} = {settings[key]}")
    except ValueError:
        print(f"Expected a number, got: {argument}")

def _set_string(settings:dict, key:str, argument:str|None) -> None:
    if argument is None:
        print(f"{key} = {settings[key]}")
        return
    settings[key] = argument
    print(f"{key} = {settings[key]}")

def _print_settings(settings:dict) -> None:
    for key, value in settings.items():
        print(f"  {key} = {value!r}")

def _print_help() -> None:
    print("commands:")
    print("  :stop-at <string>  - Sets the output sequence to stop on.")
    print("  :stop-count <int>  - Sets the number of stop sequences before stopping.")
    print("  :token-count <int> - Sets the maximum number of tokens to generate.")
    print("  :temp <float>      - Sets the sampling temperature.")
    print("  :top-k <int|off>   - Restrict sampling to the top-k tokens (off to disable).")
    print("  :show              - Shows the current settings.")
    print("  :help              - Shows this help.")
    print("  :quit              - Exits.")
