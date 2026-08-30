#!/usr/bin/env python3
import pydantic

from abc import ABC, abstractmethod
from typing import Any


class AiAgent(ABC):
    def ready(self) -> bool:
        return True

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str | None = None, skills: list[str] | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_proto(
        self,
        output_proto_model: pydantic.BaseModel,
        output_proto_class: type,
        prompt: str,
        system_instruction: str | None = None,
        skills: list[str] | None = None) -> Any:
        raise NotImplementedError


