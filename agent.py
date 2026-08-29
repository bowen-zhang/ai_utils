#!/usr/bin/env python3
import json
import time
from abc import ABC, abstractmethod
from urllib import error, request


class AiAgent(ABC):
    def ready(self) -> bool:
        return True

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str | None = None, skills: list[str] | None = None) -> str:
        raise NotImplementedError


