# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from __future__ import annotations

from bignote_gpt.training_reporter import TrainingReporter

class TensorboardTrainingReporter(TrainingReporter):
    def __init__(self, config_path:str, log_dir:str|None=None):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError(
                "TensorBoard logging requires the 'tensorboard' package. "
                "Install it with: pip install .[tensorboard]") from error
        self.__writer            = SummaryWriter(log_dir=log_dir)
        self.__config_path       = config_path
        self.__seed:int|None     = None
        self.__device_type       = ""
        self.__corpus_path       = ""
        self.__tokenizer_type    = ""
        self.__training_weight   = 0.0
        self.__validation_weight = 0.0
        self.__testing_weight    = 0.0
        self.__model_type        = ""
        self.__compiled_model    = False
        self.__batch_size        = 0
        self.__iteration_limit   = 0
        self.__evaluation_period = 0
        self.__patience          = 0
        self.__optimizer_type    = ""
        self.__lost_patience     = False
        self.__finish_iteration  = 0
        self.__trained_iteration = 0
        self.__validation_loss   = 0.0
        self.__testing_loss      = 0.0
        self.__checkpoint_path   = ""

    def on_training(self) -> None:
        pass

    def loading_config(self, config_path:str) -> None:
        pass

    def report_seed(self, seed:int) -> None:
        self.__seed = seed

    def report_device(self, device_type:str) -> None:
        self.__device_type = device_type

    def loading_corpus(self, corpus_path:str) -> None:
        self.__corpus_path = corpus_path

    def creating_tokenizer(self, tokenizer_type:str) -> None:
        self.__tokenizer_type = tokenizer_type

    def creating_datasets(self, training_weight:int, validation_weight:int, testing_weight:int) -> None:
        self.__training_weight   = training_weight
        self.__validation_weight = validation_weight
        self.__testing_weight    = testing_weight

    def creating_model(self, model_type:str) -> None:
        self.__model_type = model_type

    def compiling_model(self) -> None:
        self.__compiled_model = True

    def creating_optimizer(self, optimizer_type:str) -> None:
        self.__optimizer_type = optimizer_type

    def report_config(self, batch_size:int, iteration_limit:int, evaluation_period:int, patience:int) -> None:
        self.__batch_size        = batch_size
        self.__iteration_limit   = iteration_limit
        self.__evaluation_period = evaluation_period
        self.__patience          = patience

    def on_start_training(self):
        train, val, test = (self.__training_weight, self.__validation_weight, self.__testing_weight)
        self.__writer.add_text(
            "config",
            " |                                   |                            |\n"
            " |-----------------------------------|----------------------------|\n"
            f"| Configuration path                | {self.__config_path}       |\n"
            f"| Seed                              | {self.__seed}              |\n"
            f"| Device                            | {self.__device_type}       |\n"
            f"| Corpus path                       | {self.__corpus_path}       |\n"
            f"| Tokenizer                         | {self.__tokenizer_type}    |\n"
            f"| Training:Validation:Testing split | {train}:{val}:{test}       |\n"
            f"| Model                             | {self.__model_type}        |\n"
            f"| Compiled                          | {self.__compiled_model}    |\n"
            f"| Optimizer                         | {self.__optimizer_type}    |\n"
            f"| Batch Size                        | {self.__batch_size}        |\n"
            f"| Iteration Limit                   | {self.__iteration_limit}   |\n"
            f"| Evaluation Period                 | {self.__evaluation_period} |\n"
            f"| Patience                          | {self.__patience}          |")

    def on_evaluation(
        self,
        iteration      : int,
        learning_rate  : float,
        training_loss  : float,
        validation_loss: float,
        is_best        : bool) -> None:
        self.__writer.add_scalar("learning_rate"  , learning_rate  , iteration)
        self.__writer.add_scalars("loss", {"train": training_loss, "validation": validation_loss}, iteration)

    def on_lost_patience(self) -> None:
        self.__lost_patience = True

    def on_finish_training(self, iteration:int, trained_iteration:int, validation_loss:float) -> None:
        self.__finish_iteration  = iteration
        self.__trained_iteration = trained_iteration
        self.__validation_loss   = validation_loss

    def report_testing_loss(self, testing_loss:float) -> None:
        self.__testing_loss = testing_loss

    def saving_checkpoint(self, checkpoint_path:str) -> None:
        self.__checkpoint_path = checkpoint_path

    def on_done(self) -> None:
        self.__writer.add_text(
            "results",
            "|                       |                              |\n"
            "|-----------------------|------------------------------|\n"
            f"| Lost patience         | {self.__lost_patience}       |\n"
            f"| Finished on iteration | {self.__finish_iteration}    |\n"
            f"| Trained iteration     | {self.__trained_iteration}   |\n"
            f"| Validation loss       | {self.__validation_loss:.4f} |\n"
            f"| Testing loss          | {self.__testing_loss:.4f}    |\n"
            f"| Checkpoint path       | {self.__checkpoint_path}     | ")
        self.__writer.flush()
        self.__writer.close()
