#!/usr/bin/env python3
import pydantic

from . import agent

from google import genai
from typing import Any

class Gemini(agent.AiAgent):
    def __init__(self, api_key: str, model: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, system_instruction: str | None = None, skills: list[str] | None = None) -> str:
        contents = self._build_contents(prompt, system_instruction, skills)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents                
        )
        return response.text

    def generate_proto(
        self,
        output_proto_model: pydantic.BaseModel,
        output_proto_class: type,
        prompt: str,
        system_instruction: str | None = None,
        skills: list[str] | None = None) -> Any:
        contents = self._build_contents(prompt, system_instruction, skills)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=output_proto_model,
                temperature=0.2  # Lower temperature helps maintain strict adherence
            )
        )
        return output_proto_class.from_json(response.text)

    def _build_contents(self, prompt: str, system_instruction: str | None = None, skills: list[str] | None = None) -> str:
        parts = []
        if system_instruction:
            parts.append(system_instruction)
        if skills:
            skill_text = "\n".join(f"- {skill}" for skill in skills if skill)
            if skill_text:
                parts.append(f"Skills:\n{skill_text}")

        parts.append(prompt)

        return "\n\n".join(parts).strip()
