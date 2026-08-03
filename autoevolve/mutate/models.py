"""OpenAI-compatible model endpoint access and environment resolution."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

import httpx


@dataclass(frozen=True)
class ModelEndpoint:
    """A configured OpenAI-compatible chat completions endpoint."""

    base_url: str
    api_key: str | None
    model: str

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float | None = 0.7,
    ) -> str:
        """Send one chat completion request with bounded transient retries.

        Endpoints disagree on parameter names: current reasoning deployments
        require max_completion_tokens and reject non-default temperature,
        while older local engines only know max_tokens. The request adapts on
        a 400 that names the offending parameter, at most once per parameter.
        """

        from autoevolve.mutate.base import OperatorError

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        transient_attempts = 0
        adaptations = 0
        with httpx.Client(timeout=120.0) as client:
            while True:
                try:
                    response = client.post(url, json=payload, headers=headers)
                except httpx.HTTPError as exc:
                    raise OperatorError(f"model request failed: {exc}") from exc

                if response.is_success:
                    return _response_content(response)

                body = response.text.replace("\n", " ")[:500]
                if response.status_code == 400 and adaptations < 2:
                    if "max_completion_tokens" in body and "max_completion_tokens" in payload:
                        payload.pop("max_completion_tokens")
                        payload["max_tokens"] = max_tokens
                        adaptations += 1
                        continue
                    if "temperature" in body and "temperature" in payload:
                        payload.pop("temperature")
                        adaptations += 1
                        continue

                retryable = response.status_code == 429 or response.status_code >= 500
                if retryable and transient_attempts < 2:
                    time.sleep(min(2**transient_attempts, 4))
                    transient_attempts += 1
                    continue
                raise OperatorError(
                    f"model endpoint returned HTTP {response.status_code}: {body}"
                )


def _response_content(response: httpx.Response) -> str:
    from autoevolve.mutate.base import OperatorError

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise OperatorError("model endpoint returned an invalid chat completion response") from exc
    if not isinstance(content, str):
        raise OperatorError("model endpoint returned non-text chat completion content")
    return content


def resolve_endpoint(tier: Literal["cheap", "strong"]) -> ModelEndpoint | None:
    """Resolve local first, then explicitly configured cloud model access."""

    if tier not in {"cheap", "strong"}:
        raise ValueError(f"unknown endpoint tier: {tier}")

    local_base_url = os.getenv("AUTOEVOLVE_LOCAL_BASE_URL")
    if local_base_url:
        return ModelEndpoint(
            base_url=local_base_url,
            api_key=None,
            model=os.getenv("AUTOEVOLVE_LOCAL_MODEL", "local"),
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    tier_model = os.getenv(f"AUTOEVOLVE_MODEL_{tier.upper()}")
    model = tier_model or os.getenv("AUTOEVOLVE_MODEL")
    if not model:
        return None
    return ModelEndpoint(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
        model=model,
    )
