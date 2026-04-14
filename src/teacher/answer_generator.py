"""
src/teacher/answer_generator.py
Pluggable answer generation backends for Stage 2 (RAG) and Stage 3 (GraphRAG).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AnswerResult:
    answer: str
    model_name: str


class BaseAnswerGenerator:
    model_name: str = "unknown"

    def generate_answer(self, prompt: str) -> AnswerResult:
        raise NotImplementedError


class OpenAIAnswerGenerator(BaseAnswerGenerator):
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Install it with `pip install openai`."
            ) from exc

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI answer generator.")

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate_answer(self, prompt: str) -> AnswerResult:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        answer = (response.output_text or "").strip()
        return AnswerResult(answer=answer, model_name=self.model_name)


def build_answer_generator(
    backend: str = "openai",
    model_name: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> BaseAnswerGenerator:
    backend = backend.lower()

    if backend == "openai":
        return OpenAIAnswerGenerator(model_name=model_name, api_key=api_key)

    raise ValueError(f"Unsupported answer generator backend: {backend}")
