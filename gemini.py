#!/usr/bin/env python3
import json

from . import agent
from urllib import error, request


class Gemini(agent.AiAgent):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system_instruction: str | None = None, skills: list[str] | None = None) -> str:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

        instruction_parts = []
        if system_instruction:
            instruction_parts.append(system_instruction)
        if skills:
            skill_text = "\n".join(f"- {skill}" for skill in skills if skill)
            if skill_text:
                instruction_parts.append(f"Skills:\n{skill_text}")

        combined_instruction = "\n\n".join(instruction_parts).strip() if instruction_parts else None

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7},
        }
        if combined_instruction:
            payload["system_instruction"] = {"parts": [{"text": combined_instruction}]}
        data = json.dumps(payload).encode("utf-8")

        req = request.Request(
            f"{endpoint}?key={self.api_key}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API request failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc

        result = json.loads(body)
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response: {body}") from exc
