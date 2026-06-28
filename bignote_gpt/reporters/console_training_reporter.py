# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from __future__ import annotations

from bignote_gpt.training_reporter import TrainingReporter

class ConsoleTrainingReporter(TrainingReporter):
    def __init__(self):
        self.__printing_table = False

    def on_training(self) -> None:
        print("--- Training ---")

    def loading_config(self, config_path:str) -> None:
        print(f"Loading configuration from \"{config_path}\".")

    def report_seed(self, seed:int) -> None:
        print(f"Setting random seed to {seed}.")

    def report_device(self, device_type:str) -> None:
        print(f"Device is \"{device_type}\".")

    def loading_corpus(self, corpus_path:str) -> None:
        print(f"Loading corpus from \"{corpus_path}\".")

    def creating_tokenizer(self, tokenizer_type:str) -> None:
        print(f"Creating tokenizer \"{tokenizer_type}\".")

    def creating_datasets(self, training_weight:int, validation_weight:int, testing_weight:int) -> None:
        print(f"Creating datasets with {training_weight}:{validation_weight}:{testing_weight} "
            "split (training:validation:testing).")

    def creating_model(self, model_type:str) -> None:
        print(f"Creating model \"{model_type}\".")

    def compiling_model(self) -> None:
        print("Compiling model.")

    def creating_optimizer(self, optimizer_type:str) -> None:
        print(f"Creating optimizer \"{optimizer_type}\".")

    def report_config(self, batch_size:int, iteration_limit:int, evaluation_period:int, patience:int) -> None:
        print(f"Batch size is {batch_size}.")
        patience_text = (" or if validation loss stops improving for {patience} evaluation periods"
            if patience > 0 else "")
        print(f"Training to iteration {iteration_limit}.")
        print(f"Losses evaluated every {evaluation_period} iterations.")
        if patience > 0:
            print(f"Early stop if validation loss stops improving for {patience} evaluation periods.")

    def on_start_training(self):
        print("Training started.")
        self.__begin_table()

    def on_evaluation(
        self,
        iteration      : int,
        learning_rate  : float,
        training_loss  : float,
        validation_loss: float,
        is_best        : bool) -> None:
        best = "   *   " if is_best else "       "
        print(
            f"|     {iteration:5d} "
            f"|   {learning_rate:.2e} "
            f"|     {training_loss:.4f} "
            f"|     {validation_loss:.4f} "
            f"|{best}|")

    def on_lost_patience(self) -> None:
        self.__end_table()
        print("Validation loss stopped improving. Stopping early.")

    def on_finish_training(self, iteration:int, trained_iteration:int, validation_loss:float) -> None:
        self.__end_table()
        print(f"Training finished at iteration {iteration}.")
        print(f"Best validation loss is {validation_loss:.4f} at iteration {trained_iteration}.")

    def report_testing_loss(self, testing_loss:float) -> None:
        print(f"Testing loss is {testing_loss:.4f}.")

    def saving_checkpoint(self, checkpoint_path:str) -> None:
        print(f"Saving checkpoint to \"{checkpoint_path}\".")

    def on_done(self) -> None:
        print("--- Done ---")

    def __begin_table(self) -> None:
        print("------------------------------------------------------------")
        print("|           |  Learning  |  Training  | Validation | Best? |")
        print("| Iteration |    Rate    |    Loss    |    Loss    |  (*)  |")
        print("------------------------------------------------------------")
        self.__printing_table = True

    def __end_table(self) -> None:
        if not self.__printing_table:
            return
        print("------------------------------------------------------------")
        self.__printing_table = False
