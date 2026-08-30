
import pydantic
import time

from typing import Any

from . import agent


class RateLimiter(agent.AiAgent):
    def __init__(self, inner: agent.AiAgent, max_request_per_minute: float | None = None):
        self._inner = inner
        self._max_request_per_minute = max_request_per_minute
        self._last_request_time = None

    def ready(self) -> bool:
        return self._acquire_rate_limit(block=False)
    
    def generate(self, prompt: str, system_instruction: str | None = None, skills: list[str] | None = None) -> str:
        self._acquire_rate_limit(block=True)
        return self._inner.generate(prompt, system_instruction=system_instruction, skills=skills)

    def generate_proto(
        self,
        output_proto_model: pydantic.BaseModel,
        output_proto_class: type,
        prompt: str,
        system_instruction: str | None = None,
        skills: list[str] | None = None) -> Any:
        self._acquire_rate_limit(block=True)
        return self._inner.generate_proto(
            output_proto_model=output_proto_model,
            output_proto_class=output_proto_class,
            prompt=prompt,
            system_instruction=system_instruction,
            skills=skills
        )

    def _acquire_rate_limit(self, block: bool = True) -> bool:
        if self._max_request_per_minute is None or self._max_request_per_minute <= 0:
            return True

        if self._last_request_time is not None:
            elapsed = time.monotonic() - self._last_request_time
            min_interval = 60.0 / self._max_request_per_minute
            if elapsed < min_interval:
                if block:
                    time.sleep(min_interval - elapsed)
                else:
                    return False

        self._last_request_time = time.monotonic()
        return True


