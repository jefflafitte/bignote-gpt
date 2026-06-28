# SPDX-FileCopyrightText: 2026 Jeff Lafitte
# SPDX-License-Identifier: MIT

class Factory:
    def __init__(self, registry:dict[str, type]):
        self.__registry = registry

    def register(self, name:str, cls:type) -> None:
        if name in self.__registry:
            raise ValueError(f"Type already registered: {name}")
        self.__registry[name] = cls

    def create(self, name, *args):
        if name not in self.__registry:
            raise ValueError(f"Unknown type: {name}")
        return self.__registry[name].create(*args)
