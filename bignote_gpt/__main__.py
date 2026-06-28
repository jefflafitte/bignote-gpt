# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

import argparse

from pathlib import Path

from bignote_gpt import describe
from bignote_gpt import generate
from bignote_gpt import interact
from bignote_gpt import train
from bignote_gpt.defaults import Defaults

def _main() -> None:
    parser = argparse.ArgumentParser(
        prog        = "bignote_gpt",
        description = "BigNote GPT - A beginner-friendly generative AI harness.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="Create a training configuration file.")
    config_parser.set_defaults(apply=_config)
    config_parser.add_argument("output"   , type=str                                       , help=Defaults.help.config_path )
    config_parser.add_argument("--device" , type=_device_type, default=Defaults.device     , help=Defaults.help.device      )
    config_parser.add_argument("--compile", action="store_true"                            , help=Defaults.help.cuda_compile)
    config_parser.add_argument("--corpus" , type=str         , default=Defaults.corpus_path, help=Defaults.help.corpus_path )
    config_parser.add_argument("--model"  , type=str         , default=Defaults.model_path , help=Defaults.help.model_path  )

    describe_parser = subparsers.add_parser("describe", help="Describe a trained model.")
    describe_parser.set_defaults(apply=_describe)
    describe_parser.add_argument("model", type=_existing_path_type, help=Defaults.help.model_path)

    train_parser = subparsers.add_parser("train", help="Train a new model.")
    train_parser.set_defaults(apply=_train)
    train_parser.add_argument("config", type=_existing_path_type, help=Defaults.help.config_path)

    generate_parser = subparsers.add_parser("generate", help="Generate text using a trained model.")
    generate_parser.set_defaults(apply=_generate)
    generate_parser.add_argument("model"               , type=_existing_path_type                              , help=Defaults.help.model_path        )
    generate_parser.add_argument("--seed"              , type=int                , default=Defaults.seed       , help=Defaults.help.seed              )
    generate_parser.add_argument("--deterministic-cuda", action="store_true"                                   , help=Defaults.help.deterministic_cuda)
    generate_parser.add_argument("--prompt"            , type=str                , default=Defaults.prompt     , help=Defaults.help.prompt            )
    generate_parser.add_argument("--stop-at"           , type=str                , default=Defaults.stop_at    , help=Defaults.help.stop_at           )
    generate_parser.add_argument("--stop-count"        , type=_positive_type     , default=Defaults.stop_count , help=Defaults.help.stop_count        )
    generate_parser.add_argument("--token-count"       , type=_positive_type     , default=Defaults.token_count, help=Defaults.help.token_count       )
    generate_parser.add_argument("--temp"              , type=_temperature_type  , default=Defaults.temp       , help=Defaults.help.temp              )
    generate_parser.add_argument("--top-k"             , type=_positive_type     , default=Defaults.top_k      , help=Defaults.help.top_k             )
    generate_parser.add_argument("--device"            , type=_device_type       , default=Defaults.device     , help=Defaults.help.device            )

    interact_parser = subparsers.add_parser("interact", help="Interactively generate text from a trained model.")
    interact_parser.set_defaults(apply=_interact)
    interact_parser.add_argument("model"               , type=_existing_path_type                              , help=Defaults.help.model_path        )
    interact_parser.add_argument("--seed"              , type=int                , default=Defaults.seed       , help=Defaults.help.seed              )
    interact_parser.add_argument("--deterministic-cuda", action="store_true"                                   , help=Defaults.help.deterministic_cuda)
    interact_parser.add_argument("--stop-at"           , type=str                , default=Defaults.stop_at    , help=Defaults.help.stop_at           )
    interact_parser.add_argument("--stop-count"        , type=_positive_type     , default=Defaults.stop_count , help=Defaults.help.stop_count        )
    interact_parser.add_argument("--token-count"       , type=_positive_type     , default=Defaults.token_count, help=Defaults.help.token_count       )
    interact_parser.add_argument("--temp"              , type=_temperature_type  , default=Defaults.temp       , help=Defaults.help.temp              )
    interact_parser.add_argument("--top-k"             , type=_positive_type     , default=Defaults.top_k      , help=Defaults.help.top_k             )
    interact_parser.add_argument("--device"            , type=_device_type       , default=Defaults.device     , help=Defaults.help.device            )

    args = parser.parse_args()

    args.apply(args)

def _existing_path_type(value:str) -> str:
    if not Path(value).is_file():
        raise argparse.ArgumentTypeError(f"'{value}' does not exist.")
    return value

def _device_type(value:str) -> str:
    if value not in ("cpu", "cuda"):
        raise argparse.ArgumentTypeError("Must be \"cpu\" or \"cuda\".")
    return value

def _positive_type(value:str) -> int:
    try:
        i = int(value)
        if i <= 0:
            raise ValueError()
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return i

def _temperature_type(value:str) -> float:
    try:
        temperature = float(value)
        if (temperature <= 0) or (temperature > 2.0):
            raise ValueError()
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a number between 0.0 and 2.0.")
    return temperature

def _config(args:argparse.Namespace) -> None:
    train.config(
        device       = args.device,
        cuda_compile = args.compile,
        corpus_path  = Path(args.corpus),
        model_path   = Path(args.model),
        output_path  = Path(args.output))

def _describe(args:argparse.Namespace) -> None:
    describe.describe(checkpoint_path = Path(args.model))

def _train(args:argparse.Namespace) -> None:
    train.train(config_path=Path(args.config))

def _generate(args:argparse.Namespace) -> None:
    generate.generate(
        checkpoint_path    = Path(args.model),
        seed               = args.seed,
        deterministic_cuda = args.deterministic_cuda,
        prompt             = args.prompt,
        stop_sequence      = args.stop_at,
        stop_count         = args.stop_count,
        max_token_count    = args.token_count,
        temperature        = args.temp,
        top_k              = args.top_k,
        device_name        = args.device)

def _interact(args:argparse.Namespace) -> None:
    interact.interact(
        checkpoint_path    = Path(args.model),
        seed               = args.seed,
        deterministic_cuda = args.deterministic_cuda,
        stop_sequence      = args.stop_at,
        stop_count         = args.stop_count,
        max_token_count    = args.token_count,
        temperature        = args.temp,
        top_k              = args.top_k,
        device_name        = args.device)

if __name__ == "__main__":
    _main()
