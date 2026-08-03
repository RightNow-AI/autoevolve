import json
import random
from pathlib import Path

import httpx
import pytest

from autoevolve.core.types import Budget, Contract, EvalOutcome, ParentBundle, Program
from autoevolve.mutate import models
from autoevolve.mutate.base import OperatorContext, OperatorError
from autoevolve.mutate.diff import DiffOperator
from autoevolve.mutate.models import ModelEndpoint, resolve_endpoint


def _clear_endpoint_env(monkeypatch):
    for name in (
        "AUTOEVOLVE_LOCAL_BASE_URL",
        "AUTOEVOLVE_LOCAL_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "AUTOEVOLVE_MODEL_CHEAP",
        "AUTOEVOLVE_MODEL_STRONG",
        "AUTOEVOLVE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_resolve_endpoint_prefers_local(monkeypatch):
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("AUTOEVOLVE_LOCAL_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("AUTOEVOLVE_LOCAL_MODEL", "local-model")
    monkeypatch.setenv("OPENAI_API_KEY", "cloud-key")
    monkeypatch.setenv("AUTOEVOLVE_MODEL_CHEAP", "cloud-model")

    endpoint = resolve_endpoint("cheap")

    assert endpoint == ModelEndpoint("http://localhost:8000/v1", None, "local-model")


def test_resolve_endpoint_uses_tier_cloud_model_and_fallback(monkeypatch):
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("AUTOEVOLVE_MODEL", "fallback")
    monkeypatch.setenv("AUTOEVOLVE_MODEL_STRONG", "strong-model")

    assert resolve_endpoint("cheap") == ModelEndpoint(
        "https://example.test/v1", "secret", "fallback"
    )
    assert resolve_endpoint("strong") == ModelEndpoint(
        "https://example.test/v1", "secret", "strong-model"
    )


def test_resolve_endpoint_returns_none_when_cloud_model_is_not_named(monkeypatch):
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    assert resolve_endpoint("cheap") is None
    monkeypatch.delenv("OPENAI_API_KEY")
    assert resolve_endpoint("strong") is None


def test_chat_posts_openai_payload_and_optional_authorization(monkeypatch):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "candidate"}}]},
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    endpoint = ModelEndpoint("https://example.test/v1/", "key", "model-a")

    assert endpoint.chat([{"role": "user", "content": "go"}], max_tokens=12) == "candidate"
    assert requests[0].url == httpx.URL("https://example.test/v1/chat/completions")
    assert requests[0].headers["Authorization"] == "Bearer key"
    assert json.loads(requests[0].content) == {
        "model": "model-a",
        "messages": [{"role": "user", "content": "go"}],
        "max_completion_tokens": 12,
        "temperature": 0.7,
    }


def test_chat_adapts_params_on_400_naming_the_offender(monkeypatch):
    """Reasoning deployments reject max_completion_tokens swaps and temperature."""

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        bodies.append(payload)
        if "max_completion_tokens" in payload:
            return httpx.Response(
                400,
                text="Unsupported parameter: 'max_completion_tokens' is not supported."
                " Use 'max_tokens' instead.",
            )
        if "temperature" in payload:
            return httpx.Response(
                400,
                text="Unsupported value: 'temperature' does not support 0.7."
                " Only the default (1) value is supported.",
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "adapted"}}]})

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    endpoint = ModelEndpoint("https://example.test/v1", None, "model-a")

    assert endpoint.chat([{"role": "user", "content": "go"}], max_tokens=9) == "adapted"
    assert "max_completion_tokens" in bodies[0]
    assert bodies[1]["max_tokens"] == 9 and "max_completion_tokens" not in bodies[1]
    assert "temperature" not in bodies[2]


def test_chat_omits_temperature_when_none(monkeypatch):
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    endpoint = ModelEndpoint("https://example.test/v1", None, "model-a")

    assert endpoint.chat([{"role": "user", "content": "go"}], temperature=None) == "ok"
    assert "temperature" not in captured[0]


def test_chat_retries_transient_statuses_then_reports_last_failure(monkeypatch):
    statuses = iter((429, 500, 503))
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        status = next(statuses)
        return httpx.Response(status, text=f"failure {status}")

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr(models.time, "sleep", lambda seconds: None)
    endpoint = ModelEndpoint("https://example.test/v1", None, "model-a")

    with pytest.raises(OperatorError, match="HTTP 503: failure 503"):
        endpoint.chat([])
    assert calls == 3


def test_missing_endpoint_error_names_required_environment_variables(monkeypatch, tmp_path):
    _clear_endpoint_env(monkeypatch)
    program = Program("p1", "r1", None, "seed", "ref", 0, None, "now")
    bundle = ParentBundle(program, {"main.py": "# EVOLVE-BLOCK-START\nx\n# EVOLVE-BLOCK-END\n"})
    context = OperatorContext(
        contract=Contract(
            "improve",
            "general",
            "score",
            True,
            0.0,
            None,
            "correct",
            Budget(max_evals=1),
        ),
        rng=random.Random(1),
        endpoint_cheap=None,
        endpoint_strong=None,
        evaluate_locally=lambda files: EvalOutcome(True, {"score": 1.0}, 0),
        workdir=Path(tmp_path),
    )

    with pytest.raises(OperatorError) as caught:
        DiffOperator().propose(bundle, context)
    assert "AUTOEVOLVE_LOCAL_BASE_URL" in caught.value.reason
    assert "OPENAI_API_KEY" in caught.value.reason
    assert "AUTOEVOLVE_MODEL_CHEAP" in caught.value.reason


def _install_transport(monkeypatch, transport: httpx.MockTransport) -> None:
    real_client = httpx.Client

    def client_factory(**kwargs):
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(models.httpx, "Client", client_factory)
