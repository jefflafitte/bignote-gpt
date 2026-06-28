# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

from pathlib import Path

from bignote_gpt.dataset import Dataset
from bignote_gpt.tokenizer import Tokenizer

class Corpus:
    def __init__(self, path:Path):
        self.__path = path
        self.__text = ""

    def get_datasets(
        self,
        training_weight  : int,
        validation_weight: int,
        testing_weight   : int,
        tokenizer        : Tokenizer) -> tuple[Dataset, Dataset, Dataset]:
        if len(self.__text) == 0:
            self.__load()

        count = len(self.__text)

        weight_sum = training_weight + validation_weight + testing_weight

        training_ratio, validation_ratio = training_weight/weight_sum, validation_weight/weight_sum

        training_end, validation_end = int(training_ratio*count), int((training_ratio + validation_ratio)*count)

        training_data, validation_data, testing_data = (
            Dataset(self.__text[              :training_end  ], tokenizer),
            Dataset(self.__text[training_end  :validation_end], tokenizer),
            Dataset(self.__text[validation_end:              ], tokenizer))

        return training_data, validation_data, testing_data

    def __load(self) -> None:
        if self.__path.is_file():
            self.__text = self.__path.read_text(encoding="utf-8", errors="replace")
