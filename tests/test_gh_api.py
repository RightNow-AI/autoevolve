from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from autoevolve.gh.api import GhApiError, GitHubClient


def test_client_sends_token_auth_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"number": 7})

    with _client(handler) as client:
        assert client.get_issue(7)["number"] == 7

    assert seen[0].headers["authorization"] == "Bearer secret-token"
    assert seen[0].url.path == "/repos/acme/demo/issues/7"


def test_client_retries_500_then_succeeds() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, text="temporary")
        return httpx.Response(200, json={"number": 7})

    with _client(handler, sleep=sleeps.append) as client:
        assert client.get_issue(7)["number"] == 7

    assert calls == 2
    assert sleeps == [1.0]


def test_client_honors_retry_after_for_rate_limit() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                403,
                text="API rate limit exceeded",
                headers={"Retry-After": "3"},
            )
        return httpx.Response(200, json={"number": 7})

    with _client(handler, sleep=sleeps.append) as client:
        client.get_issue(7)

    assert sleeps == [3.0]


def test_api_error_carries_status_and_bounded_body_excerpt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="missing " + "x" * 600)

    with _client(handler) as client, pytest.raises(GhApiError) as raised:
        client.get_issue(99)

    assert raised.value.status == 404
    assert raised.value.body_excerpt.startswith("missing")
    assert len(raised.value.body_excerpt) == 500


def test_put_files_builds_blobs_tree_commit_and_ref_in_order() -> None:
    calls: list[tuple[str, str, Any]] = []
    blob_index = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal blob_index
        payload = _json(request)
        calls.append((request.method, request.url.path, payload))
        path = request.url.path
        if path.endswith("/git/refs"):
            return httpx.Response(201, json={"ref": "refs/heads/autoevolve/run-r1"})
        if path.endswith("/git/commits/base-sha") and request.method == "GET":
            return httpx.Response(200, json={"sha": "base-sha", "tree": {"sha": "tree-0"}})
        if path.endswith("/git/blobs"):
            blob_index += 1
            return httpx.Response(201, json={"sha": f"blob-{blob_index}"})
        if path.endswith("/git/trees"):
            return httpx.Response(201, json={"sha": "tree-1"})
        if path.endswith("/git/commits"):
            return httpx.Response(201, json={"sha": "commit-1"})
        if "/git/refs/heads/" in path and request.method == "PATCH":
            return httpx.Response(200, json={"object": {"sha": "commit-1"}})
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    with _client(handler) as client:
        client.create_branch("base-sha", "autoevolve/run-r1")
        commit = client.put_files(
            "autoevolve/run-r1",
            {"src/a.py": "a = 1\n", "assets/image.bin": b"\x00\x01"},
            "Add result",
        )

    assert commit["sha"] == "commit-1"
    operations = [(method, path.rsplit("/", 1)[-1]) for method, path, _ in calls]
    assert operations == [
        ("POST", "refs"),
        ("GET", "base-sha"),
        ("POST", "blobs"),
        ("POST", "blobs"),
        ("POST", "trees"),
        ("POST", "commits"),
        ("PATCH", "run-r1"),
    ]
    assert calls[3][2]["encoding"] == "base64"


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    sleep: Callable[[float], None] = lambda _: None,
) -> GitHubClient:
    return GitHubClient(
        "secret-token",
        "acme/demo",
        "https://api.github.test",
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )


def _json(request: httpx.Request) -> Any:
    if not request.content:
        return None
    return json.loads(request.content)
