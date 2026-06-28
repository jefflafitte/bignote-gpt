# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from pathlib import Path

from bignote_gpt.checkpoint import describe_checkpoint

def describe(checkpoint_path: Path) -> None:
    description = describe_checkpoint(checkpoint_path)

    print(f"created              : {description.timestamp}")
    print(f"seed                 : {description.seed}")
    print(f"tokenizer            : {description.tokenizer_type}")
    print(f"model                : {description.model_type}")
    print(f"parameter count      : {description.parameter_count}")
    print(f"trained iteration    : {description.trained_iteration}")
    print(f"training loss        : {description.training_loss:.2f}")
    print(f"validation loss      : {description.validation_loss:.2f}")
    print(f"testing loss         : {description.testing_loss:.2f}")
    print(f"model configuration  : {description.model_config}")
    print(f"trainer configuration: {description.trainer_config}")
