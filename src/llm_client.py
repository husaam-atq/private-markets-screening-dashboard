from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from src.config import get_env, load_config


@dataclass
class LLMResponse:
    text: str
    model_name: str
    used_llm: bool
    warning: str = ""


def _generation_config() -> dict[str, Any]:
    return load_config("rag_config.yaml").get("generation", {})


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        cfg = _generation_config()
        self.base_url = (base_url or get_env("OLLAMA_BASE_URL") or cfg.get("ollama_base_url") or "http://localhost:11434").rstrip("/")
        self.model_name = model_name or get_env("OLLAMA_MODEL") or cfg.get("ollama_model", "qwen3")
        self.timeout_seconds = int(timeout_seconds or cfg.get("request_timeout_seconds", 45))

    def is_ollama_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=min(self.timeout_seconds, 5))
            return response.ok
        except requests.RequestException:
            return False

    def available_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=min(self.timeout_seconds, 5))
            response.raise_for_status()
            payload = response.json()
            return [str(item.get("name", "")) for item in payload.get("models", [])]
        except (requests.RequestException, ValueError):
            return []

    def resolve_model_name(self, model_name: str | None = None) -> str | None:
        requested = model_name or self.model_name
        available = self.available_models()
        if requested in available:
            return requested
        requested_base = requested.split(":")[0]
        for model in available:
            if model.split(":")[0] == requested_base:
                return model
        return None

    def is_model_available(self, model_name: str | None = None) -> bool:
        return self.resolve_model_name(model_name) is not None

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 700) -> LLMResponse:
        if not self.is_ollama_available():
            return LLMResponse("", self.model_name, False, "Ollama is not available; deterministic fallback used.")
        resolved_model = self.resolve_model_name(self.model_name)
        if not resolved_model:
            return LLMResponse("", self.model_name, False, f"Model {self.model_name} is not available; deterministic fallback used.")
        payload = {
            "model": resolved_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            return LLMResponse(str(response.json().get("response", "")).strip(), resolved_model, True)
        except (requests.RequestException, ValueError) as exc:
            return LLMResponse("", self.model_name, False, f"Ollama generation failed: {type(exc).__name__}; deterministic fallback used.")


def is_llm_enabled() -> bool:
    cfg = _generation_config()
    env = str(get_env("LLM_ENABLED", "")).lower()
    if env in {"true", "1", "yes"}:
        return True
    if env in {"false", "0", "no"}:
        return False
    return bool(cfg.get("llm_enabled", False))
