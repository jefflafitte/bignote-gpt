# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from dataclasses import  dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from bignote_gpt.factories.model_factory import model_factory
from bignote_gpt.factories.tokenizer_factory import tokenizer_factory
from bignote_gpt.model import Model

@dataclass
class CheckpointDescription:
    timestamp        : datetime
    seed             : int|None
    tokenizer_type   : str
    model_type       : str
    model_config     : dict
    parameter_count  : int
    trainer_config   : dict
    trained_iteration: int
    training_loss    : float
    validation_loss  : float
    testing_loss     : float

def save_checkpoint(
    path             : Path,
    seed             : int|None,
    tokenizer_type   : str,
    model_type       : str,
    model_config     : dict,
    model            : Model,
    trainer_config   : dict,
    trained_iteration: int,
    training_loss    : float,
    validation_loss  : float,
    testing_loss     : float) -> None:
    torch.save({
        "metadata" : {
            "timestamp"        : datetime.now(timezone.utc).isoformat(),
            "seed"             : seed,
            "parameter_count"  : sum(parameter.numel() for parameter in model.parameters()),
            "trainer_config"   : trainer_config,
            "trained_iteration": trained_iteration,
            "training_loss"    : training_loss,
            "validation_loss"  : validation_loss,
            "testing_loss"     : testing_loss},
        "tokenizer": {"type": tokenizer_type},
        "model"    : {"type": model_type, "config": model_config, "state": model.state_dict()}
    }, path)

def describe_checkpoint(path:Path) -> CheckpointDescription:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    metadata       = checkpoint["metadata"]
    tokenizer_data = checkpoint["tokenizer"]
    model_data     = checkpoint["model"]

    return CheckpointDescription(
        timestamp         = datetime.fromisoformat(metadata["timestamp"]),
        seed              = metadata["seed"],
        tokenizer_type    = tokenizer_data["type"],
        model_type        = model_data["type"],
        model_config      = model_data["config"],
        parameter_count   = metadata["parameter_count"],
        trainer_config    = metadata["trainer_config"],
        trained_iteration = metadata["trained_iteration"],
        training_loss     = metadata["training_loss"],
        validation_loss   = metadata["validation_loss"],
        testing_loss      = metadata["testing_loss"])

def load_checkpoint(path:Path, device:torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    tokenizer_data = checkpoint["tokenizer"]
    tokenizer      = tokenizer_factory.create(tokenizer_data["type"])

    model_data = checkpoint["model"]
    model      = model_factory.create(model_data["type"], tokenizer.vocabulary_count, model_data["config"])

    model.load_state_dict(model_data["state"])
    model.to(device)
    model.eval()

    return tokenizer, model
