# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from __future__ import annotations

from bignote_gpt.training_reporter import TrainingReporter

class CompositeTrainingReporter(TrainingReporter):
    def __init__(self, *args:TrainingReporter):
        self.__reporters = args

    def on_training(self) -> None:
        for reporter in self.__reporters:
            reporter.on_training()

    def loading_config(self, config_path:str) -> None:
        for reporter in self.__reporters:
            reporter.loading_config(config_path)

    def report_seed(self, seed:int) -> None:
        for reporter in self.__reporters:
            reporter.report_seed(seed)

    def report_device(self, device_type:str) -> None:
        for reporter in self.__reporters:
            reporter.report_device(device_type)

    def loading_corpus(self, corpus_path:str) -> None:
        for reporter in self.__reporters:
            reporter.loading_corpus(corpus_path)

    def creating_tokenizer(self, tokenizer_type:str) -> None:
        for reporter in self.__reporters:
            reporter.creating_tokenizer(tokenizer_type)

    def creating_datasets(self, training_weight:int, validation_weight:int, testing_weight:int) -> None:
        for reporter in self.__reporters:
            reporter.creating_datasets(training_weight, validation_weight, testing_weight)

    def creating_model(self, model_type:str) -> None:
        for reporter in self.__reporters:
            reporter.creating_model(model_type)

    def compiling_model(self) -> None:
        for reporter in self.__reporters:
            reporter.compiling_model()

    def creating_optimizer(self, optimizer_type:str) -> None:
        for reporter in self.__reporters:
            reporter.creating_optimizer(optimizer_type)

    def report_config(self, batch_size:int, iteration_limit:int, evaluation_period:int, patience:int) -> None:
        for reporter in self.__reporters:
            reporter.report_config(batch_size, iteration_limit, evaluation_period, patience)

    def on_start_training(self):
        for reporter in self.__reporters:
            reporter.on_start_training()

    def on_evaluation(
        self,
        iteration      : int,
        learning_rate  : float,
        training_loss  : float,
        validation_loss: float,
        is_best        : bool) -> None:
        for reporter in self.__reporters:
            reporter.on_evaluation(iteration, learning_rate, training_loss, validation_loss, is_best)

    def on_lost_patience(self) -> None:
        for reporter in self.__reporters:
            reporter.on_lost_patience()

    def on_finish_training(self, iteration:int, trained_iteration:int, validation_loss:float) -> None:
        for reporter in self.__reporters:
            reporter.on_finish_training(iteration, trained_iteration, validation_loss)

    def report_testing_loss(self, testing_loss:float) -> None:
        for reporter in self.__reporters:
            reporter.report_testing_loss(testing_loss)

    def saving_checkpoint(self, checkpoint_path:str) -> None:
        for reporter in self.__reporters:
            reporter.saving_checkpoint(checkpoint_path)

    def on_done(self) -> None:
        for reporter in self.__reporters:
            reporter.on_done()
