# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import tomlkit

import torch

from bignote_gpt.checkpoint import save_checkpoint
from bignote_gpt.corpus import Corpus
from bignote_gpt.defaults import Defaults
from bignote_gpt.factories.model_factory import model_factory
from bignote_gpt.factories.optimizer_factory import optimizer_factory
from bignote_gpt.factories.tokenizer_factory import tokenizer_factory
from bignote_gpt.model import Model
from bignote_gpt.models.classic.gpt import GPTConfig
from bignote_gpt.optimizers.adamw_optimizer import AdamWOptimizerConfig
from bignote_gpt.reporters.composite_training_reporter import CompositeTrainingReporter
from bignote_gpt.reporters.console_training_reporter import ConsoleTrainingReporter
from bignote_gpt.reporters.tensorboard_training_reporter import TensorboardTrainingReporter
from bignote_gpt.seeding import set_seed
from bignote_gpt.trainer import Trainer, TrainingReporter
from bignote_gpt.trainer import TrainerConfig
from bignote_gpt.training_reporter import TrainingReporter

@dataclass
class SystemConfig:
    seed               : int|None = field(default=Defaults.seed               , metadata={"help": Defaults.help.seed               })
    device             : str      = field(default=Defaults.device             , metadata={"help": Defaults.help.device             })
    deterministic_cuda : bool     = field(default=Defaults.deterministic_cuda , metadata={"help": Defaults.help.deterministic_cuda })
    cuda_compile       : bool     = field(default=Defaults.cuda_compile       , metadata={"help": Defaults.help.cuda_compile       })
    tensorboard        : bool     = field(default=Defaults.tensorboard        , metadata={"help": Defaults.help.tensorboard        })
    tensorboard_log_dir: str|None = field(default=Defaults.tensorboard_log_dir, metadata={"help": Defaults.help.tensorboard_log_dir})

@dataclass
class CorpusConfig:
    path: str = field(default=Defaults.corpus_path, metadata={"help": Defaults.help.corpus_path})

@dataclass
class DatasetsConfig:
    training_weight  : int = field(default=Defaults.training_weight  , metadata={"help": Defaults.help.training_weight  })
    validation_weight: int = field(default=Defaults.validation_weight, metadata={"help": Defaults.help.validation_weight})
    testing_weight   : int = field(default=Defaults.testing_weight   , metadata={"help": Defaults.help.testing_weight   })

@dataclass
class TokenizerConfig:
    type: str = field(default=Defaults.tokenizer_type, metadata={"help": Defaults.help.tokenizer_type})

@dataclass
class ModelConfig:
    type  : str  = field(default=Defaults.model_type, metadata={"help": Defaults.help.model_type})
    config: dict = field(default_factory=lambda: asdict(GPTConfig()), metadata={"default": GPTConfig})

@dataclass
class OptimizerConfig:
    type  : str  = field(default=Defaults.optimizer_type, metadata={"help": Defaults.help.optimizer_type})
    config: dict = field(default_factory=lambda: asdict(AdamWOptimizerConfig()), metadata={"default": AdamWOptimizerConfig})

@dataclass
class OutputConfig:
    path: str = field(default=Defaults.model_path, metadata={"help": Defaults.help.model_path})

@dataclass
class TrainConfig:
    system   : SystemConfig    = field(default_factory=SystemConfig)
    corpus   : CorpusConfig    = field(default_factory=CorpusConfig)
    datasets : DatasetsConfig  = field(default_factory=DatasetsConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model    : ModelConfig     = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer  : TrainerConfig   = field(default_factory=TrainerConfig)
    output   : OutputConfig    = field(default_factory=OutputConfig)

def config(device:str, cuda_compile:bool, corpus_path:Path, model_path:Path, output_path:Path) -> None:
    train_config                     = TrainConfig()
    train_config.system.device       = device if device == "cuda" else train_config.system.device
    train_config.system.cuda_compile = cuda_compile and (train_config.system.device == "cuda")
    train_config.corpus.path         = corpus_path.as_posix()
    train_config.output.path         = model_path.as_posix()

    document = tomlkit.document()

    def make_table(instance:Any, is_config:bool=False) -> tomlkit.items.Table:
        table = tomlkit.table()
        first_item = True
        for field in fields(instance):
            if is_config or (field.name != "config"):
                if first_item:
                    first_item = False
                else:
                   table.add(tomlkit.nl())
                comment = field.metadata.get("help")
                if comment:
                    table.add(tomlkit.comment(comment))
                value = getattr(instance, field.name)
                if value is not None:
                    if isinstance(value, tuple):
                        value = list(value)
                    table.add(field.name, value)
                else:
                    table.add(tomlkit.comment(f"{field.name} = "))
            else:
                default_class = field.metadata.get("default")
                if default_class is not None:
                    table["config"] = make_table(default_class(), True)
        return table

    for field in fields(train_config):
        document[field.name] = make_table(getattr(train_config, field.name))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tomlkit.dumps(document))

def train(config_path:Path) -> None:
    reporter:TrainingReporter = ConsoleTrainingReporter()

    reporter.on_training()
    reporter.loading_config(config_path=config_path.as_posix())

    with config_path.open("rb") as config_file:
        config_data = tomlkit.load(config_file).unwrap()

    config = TrainConfig(
        system    = SystemConfig   (**config_data["system"   ]),
        corpus    = CorpusConfig   (**config_data["corpus"   ]),
        datasets  = DatasetsConfig (**config_data["datasets" ]),
        tokenizer = TokenizerConfig(**config_data["tokenizer"]),
        model     = ModelConfig    (  config_data["model"    ]["type"], config_data["model"]    .get("config", {})),
        optimizer = OptimizerConfig(  config_data["optimizer"]["type"], config_data["optimizer"].get("config", {})),
        trainer   = TrainerConfig  (**config_data["trainer"  ]),
        output    = OutputConfig   (**config_data["output"   ]))

    cuda_requested = (config.system.device == "cuda")
    using_cuda     = cuda_requested and torch.cuda.is_available()

    if config.system.tensorboard:
        reporter = CompositeTrainingReporter(
            reporter,
            TensorboardTrainingReporter(config_path.as_posix(), config.system.tensorboard_log_dir))

    if config.system.seed is not None:
        reporter.report_seed(config.system.seed)
        set_seed(config.system.seed, using_cuda and config.system.deterministic_cuda)

    device_type = "cuda" if using_cuda else "cpu"

    device = torch.device("cuda" if using_cuda else "cpu")

    reporter.report_device(device_type)

    torch.set_float32_matmul_precision("high")

    reporter.loading_corpus(config.corpus.path)

    corpus = Corpus(Path(config.corpus.path))

    reporter.creating_tokenizer(config.tokenizer.type)

    tokenizer = tokenizer_factory.create(config.tokenizer.type)

    reporter.creating_datasets(
        config.datasets.training_weight,
        config.datasets.validation_weight,
        config.datasets.testing_weight)

    training_data, validation_data, testing_data = corpus.get_datasets(
        config.datasets.training_weight,
        config.datasets.validation_weight,
        config.datasets.testing_weight,
        tokenizer)

    reporter.creating_model(config.model.type)

    model = model_factory.create(config.model.type, tokenizer.vocabulary_count, config.model.config)

    model.to(device)

    training_model = model

    if (using_cuda and config.system.cuda_compile):
        reporter.compiling_model()
        training_model = cast(Model, torch.compile(model))

    reporter.creating_optimizer(config.optimizer.type)

    optimizer = optimizer_factory.create(
        config.optimizer.type,
        model,
        config.trainer.learning_rate,
        config.optimizer.config)

    reporter.report_config(
        config.trainer.batch_size,
        config.trainer.iteration_limit,
        config.trainer.evaluation_period,
        config.trainer.patience)

    trainer = Trainer(
        config          = config.trainer,
        model           = training_model,
        optimizer       = optimizer,
        training_data   = training_data,
        validation_data = validation_data,
        device          = device,
        reporter        = reporter)

    training_result = trainer.train()

    testing_loss = trainer.estimate_loss(testing_data)

    reporter.report_testing_loss(testing_loss)
    reporter.saving_checkpoint(config.output.path)

    output = Path(config.output.path)
    output.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        path              = Path(config.output.path),
        seed              = config.system.seed,
        tokenizer_type    = config.tokenizer.type,
        model_type        = config.model.type,
        model_config      = config.model.config,
        model             = model,
        trainer_config    = asdict(config.trainer),
        trained_iteration = training_result.trained_iteration,
        training_loss     = training_result.training_loss,
        validation_loss   = training_result.validation_loss,
        testing_loss      = testing_loss)

    reporter.on_done()
