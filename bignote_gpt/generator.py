# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

import torch

from bignote_gpt.model import Model
from bignote_gpt.tokenizer import Tokenizer

HARD_MAX_TOKEN_COUNT = 10000

class Generator:
    def __init__(self, tokenizer:Tokenizer, model:Model, device:torch.device):
        self.__tokenizer = tokenizer
        self.__model     = model
        self.__device    = device

    @torch.no_grad()
    def generate(
        self,
        prompt         : str,
        stop_sequence  : str,
        stop_count     : int,
        max_token_count: int,
        temperature    : float    = 1.0,
        top_k          : int|None = None) -> str:
        temperature = temperature if temperature <= 0 else 1.0

        self.__model.eval()

        input = torch.tensor([self.__tokenizer.encode(prompt)], dtype=torch.long, device=self.__device)

        generation = _Generation(self.__tokenizer, stop_sequence, stop_count, max_token_count)

        while not generation.done():
            cropped = input[:, -self.__model.max_sequence_size:]

            logits, _ = self.__model(cropped)
            logits = logits[:, -1, :]/temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < values[:, [-1]]] = float("-inf")

            probabilities = torch.softmax(logits, dim=-1)
            next_token    = torch.multinomial(probabilities, num_samples=1)

            input = torch.cat((input, next_token), dim=1)

            generation.append_token(next_token)

        self.__model.train()

        return prompt + generation.get_text()

class _Generation:
    def __init__(
        self,
        tokenizer      :Tokenizer,
        stop_sequence  : str,
        stop_count     : int,
        max_token_count: int):
        self.__tokenizer         = tokenizer
        self.__stop_sequence     = stop_sequence
        self.__max_stop_count    = stop_count
        self.__max_token_count   = min(max_token_count, HARD_MAX_TOKEN_COUNT)
        self.__stop_length       = len(stop_sequence)
        self.__stop_search_index = 0
        self.__stop_count        = 0
        self.__tokens: list[int] = []

    def append_token(self, token:torch.Tensor):
        self.__tokens.append(int(token.item()))

    def done(self) -> bool:
        if len(self.__tokens) >= self.__max_token_count:
            return True

        if self.__stop_length > 0:
            while True:
                tail = self.__tokenizer.decode(self.__tokens[self.__stop_search_index:])

                stop_index = tail.find(self.__stop_sequence)

                if stop_index == -1:
                    break

                self.__stop_count += 1

                if self.__stop_count >= self.__max_stop_count:
                    return True

                self.__stop_search_index = self.__advance_stop_search_index(tail, stop_index)

        return False

    def get_text(self) -> str:
        generated_text = self.__tokenizer.decode(self.__tokens)
        end_index      = len(generated_text)

        if (self.__stop_length > 0) and (self.__stop_count >= self.__max_stop_count):
            end_index = 0
            for _ in range(self.__max_stop_count):
                stop_index = generated_text.find(self.__stop_sequence, end_index)
                if stop_index == -1:
                    end_index = len(generated_text)
                    break
                end_index = stop_index + self.__stop_length

        return generated_text[:end_index]

    def __advance_stop_search_index(self, tail:str, stop_index:int) -> int:
        for k in range(self.__stop_search_index + 1, len(self.__tokens) + 1):
            decoded = self.__tokenizer.decode(self.__tokens[self.__stop_search_index:k])
            if tail.startswith(decoded) and (len(decoded) >= (stop_index + self.__stop_length)):
                return k
        return len(self.__tokens)
