from __future__ import annotations

import requests

from src import llm_client
from src.llm_client import LLMResponse, OllamaClient, is_llm_enabled


class FakeResponse:
    def __init__(self, payload=None, ok=True, raise_exc=None):
        self._payload = payload or {}
        self.ok = ok
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self):
        return self._payload


def test_client_uses_config_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    client = OllamaClient()
    assert client.base_url == "http://localhost:11434"
    assert client.model_name == "qwen3"
    assert client.timeout_seconds == 45


def test_client_env_overrides_config(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.local:9999/")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1")
    client = OllamaClient()
    assert client.base_url == "http://example.local:9999"
    assert client.model_name == "llama3.1"


def test_is_ollama_available_true(monkeypatch):
    monkeypatch.setattr(llm_client.requests, "get", lambda *a, **k: FakeResponse(ok=True))
    assert OllamaClient().is_ollama_available() is True


def test_is_ollama_available_handles_exception(monkeypatch):
    def raise_conn(*a, **k):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(llm_client.requests, "get", raise_conn)
    assert OllamaClient().is_ollama_available() is False


def test_available_models_parses_payload(monkeypatch):
    payload = {"models": [{"name": "qwen3:latest"}, {"name": "llama3.1"}]}
    monkeypatch.setattr(llm_client.requests, "get", lambda *a, **k: FakeResponse(payload=payload))
    assert OllamaClient().available_models() == ["qwen3:latest", "llama3.1"]


def test_available_models_returns_empty_on_error(monkeypatch):
    def raise_conn(*a, **k):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(llm_client.requests, "get", raise_conn)
    assert OllamaClient().available_models() == []


def test_resolve_model_name_exact_and_base_match(monkeypatch):
    client = OllamaClient(model_name="qwen3")
    monkeypatch.setattr(client, "available_models", lambda: ["qwen3:latest", "llama3.1"])
    # base match: requested "qwen3" resolves to "qwen3:latest"
    assert client.resolve_model_name() == "qwen3:latest"
    # exact match wins
    monkeypatch.setattr(client, "available_models", lambda: ["qwen3", "qwen3:latest"])
    assert client.resolve_model_name("qwen3") == "qwen3"


def test_resolve_model_name_returns_none_when_missing(monkeypatch):
    client = OllamaClient(model_name="does-not-exist")
    monkeypatch.setattr(client, "available_models", lambda: ["llama3.1"])
    assert client.resolve_model_name() is None
    assert client.is_model_available() is False


def test_generate_returns_fallback_when_unavailable(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(client, "is_ollama_available", lambda: False)
    response = client.generate("prompt")
    assert isinstance(response, LLMResponse)
    assert response.used_llm is False
    assert "not available" in response.warning


def test_generate_returns_fallback_when_model_missing(monkeypatch):
    client = OllamaClient(model_name="ghost")
    monkeypatch.setattr(client, "is_ollama_available", lambda: True)
    monkeypatch.setattr(client, "resolve_model_name", lambda *_a: None)
    response = client.generate("prompt")
    assert response.used_llm is False
    assert "ghost" in response.warning


def test_generate_success(monkeypatch):
    client = OllamaClient(model_name="qwen3")
    monkeypatch.setattr(client, "is_ollama_available", lambda: True)
    monkeypatch.setattr(client, "resolve_model_name", lambda *_a: "qwen3:latest")
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        lambda *a, **k: FakeResponse(payload={"response": "  grounded answer  "}),
    )
    response = client.generate("prompt", temperature=0.2, max_tokens=100)
    assert response.used_llm is True
    assert response.text == "grounded answer"
    assert response.model_name == "qwen3:latest"


def test_generate_handles_request_error(monkeypatch):
    client = OllamaClient(model_name="qwen3")
    monkeypatch.setattr(client, "is_ollama_available", lambda: True)
    monkeypatch.setattr(client, "resolve_model_name", lambda *_a: "qwen3:latest")

    def raise_post(*a, **k):
        raise requests.Timeout("slow")

    monkeypatch.setattr(llm_client.requests, "post", raise_post)
    response = client.generate("prompt")
    assert response.used_llm is False
    assert "Timeout" in response.warning


def test_is_llm_enabled_env_true(monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "true")
    assert is_llm_enabled() is True


def test_is_llm_enabled_env_false(monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "no")
    assert is_llm_enabled() is False


def test_is_llm_enabled_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("LLM_ENABLED", raising=False)
    # config default is llm_enabled: false
    assert is_llm_enabled() is False
