from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ChatModel(Protocol):
    model: str

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...
class EmbeddingModel(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...




class ModelClientError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleChatModel:
    """Small stdlib client for vLLM and other OpenAI-compatible servers."""

    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    timeout_seconds: float = 60.0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._complete(system_prompt, user_prompt, json_mode=False)

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        """Request an API-enforced JSON object from compatible chat servers."""
        return self._complete(system_prompt, user_prompt, json_mode=True)

    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool,
    ) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if json_mode:
            request_body["response_format"] = {"type": "json_object"}
        payload = json.dumps(request_body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise ModelClientError(f"model HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ModelClientError(f"model request failed: {error}") from error
        try:
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ModelClientError("model response content is empty")
            return content
        except (KeyError, IndexError, TypeError) as error:
            raise ModelClientError("unexpected model response shape") from error


@dataclass
class OpenAICompatibleEmbeddingModel:
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 60.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        endpoint = f"{self.base_url.rstrip('/')}/embeddings"
        payload = json.dumps(
            {"model": self.model, "input": texts}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise ModelClientError(
                f"embedding HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ModelClientError(f"embedding request failed: {error}") from error
        try:
            records = sorted(data["data"], key=lambda item: int(item["index"]))
            vectors = [
                [float(value) for value in item["embedding"]]
                for item in records
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ModelClientError(
                "unexpected embedding response shape"
            ) from error
        if len(vectors) != len(texts):
            raise ModelClientError(
                f"embedding count mismatch: expected={len(texts)}, got={len(vectors)}"
            )
        if any(not vector for vector in vectors):
            raise ModelClientError("embedding response contains an empty vector")
        return vectors


def build_chat_model(config: dict[str, object]) -> ChatModel | None:
    provider = str(config.get("provider", "heuristic"))
    if provider == "heuristic":
        return None
    if provider != "openai_compatible":
        raise ValueError(f"지원하지 않는 model provider: {provider}")
    return OpenAICompatibleChatModel(
        base_url=str(config.get("base_url", "http://localhost:8000/v1")),
        model=str(config.get("model", "Qwen/Qwen3-4B-Instruct-2507")),
        api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
        temperature=float(config.get("temperature", 0.0)),
        timeout_seconds=float(config.get("timeout_seconds", 60.0)),
    )
def build_embedding_model(config: dict[str, Any]) -> EmbeddingModel:
    provider = str(config.get("provider", "openai_compatible"))
    if provider != "openai_compatible":
        raise ValueError("embedding provider는 openai_compatible만 지원합니다.")
    return OpenAICompatibleEmbeddingModel(
        base_url=str(config.get("base_url", "http://localhost:8001/v1")),
        model=str(config.get("model", "Qwen/Qwen3-Embedding-0.6B")),
        api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
        timeout_seconds=float(config.get("timeout_seconds", 60.0)),
    )
