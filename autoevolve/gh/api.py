"""Small, typed GitHub REST client used by issue mode."""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import httpx

FileContent = str | bytes


class GhApiError(RuntimeError):
    """A non-success GitHub API response with a bounded body excerpt."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.status_code = status
        self.message = message
        self.body_excerpt = message[:500]
        super().__init__(f"GitHub API returned HTTP {status}: {self.body_excerpt}")


class GitHubClient:
    """The GitHub operations required by issue mode."""

    def __init__(
        self,
        token: str,
        repository: str,
        api_url: str = "https://api.github.com",
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not token:
            raise ValueError("A GitHub token is required.")
        repository_parts = repository.split("/")
        if len(repository_parts) != 2 or not all(repository_parts):
            raise ValueError("repository must have the form owner/name")
        self.repository = repository
        self._sleep = sleep
        self._branch_heads: dict[str, str] = {}
        self._client = httpx.Client(
            base_url=f"{api_url.rstrip('/')}/",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "autoevolve-gh/0.1",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_issue(self, issue_number: int) -> dict[str, Any]:
        return self._dict_request("GET", self._repo_path(f"issues/{issue_number}"))

    def post_comment(self, issue_number: int, body: str) -> dict[str, Any]:
        return self._dict_request(
            "POST",
            self._repo_path(f"issues/{issue_number}/comments"),
            json={"body": body},
        )

    def list_labels(self, issue_number: int) -> list[dict[str, Any]]:
        labels: list[dict[str, Any]] = []
        page = 1
        while True:
            response = self._request(
                "GET",
                self._repo_path(f"issues/{issue_number}/labels"),
                params={"per_page": 100, "page": page},
            )
            payload = _json_payload(response)
            if not isinstance(payload, list):
                raise GhApiError(response.status_code, "Expected a list of issue labels.")
            page_labels = [item for item in payload if isinstance(item, dict)]
            labels.extend(page_labels)
            if len(payload) < 100:
                break
            page += 1
        return labels

    def get_actor_permission(self, username: str) -> dict[str, Any]:
        encoded = quote(username, safe="")
        return self._dict_request(
            "GET", self._repo_path(f"collaborators/{encoded}/permission")
        )

    def get_default_branch(self) -> dict[str, Any]:
        repository = self._dict_request("GET", self._repo_path(""))
        name = repository.get("default_branch")
        if not isinstance(name, str) or not name:
            raise GhApiError(200, "Repository response did not contain a default branch.")
        branch = self._dict_request(
            "GET", self._repo_path(f"git/ref/heads/{quote(name, safe='/')}")
        )
        sha = _nested_string(branch, "object", "sha")
        return {"name": name, "sha": sha}

    def create_branch(self, base_sha: str, name: str) -> dict[str, Any]:
        result = self._dict_request(
            "POST",
            self._repo_path("git/refs"),
            json={"ref": f"refs/heads/{name}", "sha": base_sha},
        )
        self._branch_heads[name] = base_sha
        return result

    def put_files(
        self,
        branch: str,
        files: Mapping[str, FileContent],
        message: str,
    ) -> dict[str, Any]:
        if not files:
            raise ValueError("At least one file is required for a commit.")
        base_sha = self._branch_heads.get(branch)
        if base_sha is None:
            ref = self._dict_request(
                "GET", self._repo_path(f"git/ref/heads/{quote(branch, safe='/')}")
            )
            base_sha = _nested_string(ref, "object", "sha")

        parent = self._dict_request("GET", self._repo_path(f"git/commits/{base_sha}"))
        base_tree = _nested_string(parent, "tree", "sha")
        tree_entries: list[dict[str, str]] = []
        for path, content in files.items():
            clean_path = _clean_repo_path(path)
            blob_payload = _blob_payload(content)
            blob = self._dict_request(
                "POST", self._repo_path("git/blobs"), json=blob_payload
            )
            tree_entries.append(
                {
                    "path": clean_path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": _required_string(blob, "sha"),
                }
            )

        tree = self._dict_request(
            "POST",
            self._repo_path("git/trees"),
            json={"base_tree": base_tree, "tree": tree_entries},
        )
        commit = self._dict_request(
            "POST",
            self._repo_path("git/commits"),
            json={
                "message": message,
                "tree": _required_string(tree, "sha"),
                "parents": [base_sha],
            },
        )
        commit_sha = _required_string(commit, "sha")
        self._dict_request(
            "PATCH",
            self._repo_path(f"git/refs/heads/{quote(branch, safe='/')}"),
            json={"sha": commit_sha, "force": False},
        )
        self._branch_heads[branch] = commit_sha
        return commit

    def create_pr(
        self,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> dict[str, Any]:
        return self._dict_request(
            "POST",
            self._repo_path("pulls"),
            json={"head": head, "base": base, "title": title, "body": body},
        )

    def _repo_path(self, suffix: str) -> str:
        base = f"repos/{self.repository}"
        return f"{base}/{suffix}" if suffix else base

    def _dict_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._request(method, path, **kwargs)
        payload = _json_payload(response)
        if not isinstance(payload, dict):
            raise GhApiError(response.status_code, "Expected a JSON object from GitHub.")
        return payload

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(3):
            response = self._client.request(method, path, **kwargs)
            if 200 <= response.status_code < 300:
                return response
            if attempt < 2 and _retryable(response):
                self._sleep(_retry_delay(response, attempt))
                continue
            excerpt = " ".join(response.text.split())[:500] or "empty response body"
            raise GhApiError(response.status_code, excerpt)
        raise RuntimeError("unreachable retry state")


def _retryable(response: httpx.Response) -> bool:
    if response.status_code >= 500:
        return True
    if response.status_code != 403:
        return False
    body = response.text.lower()
    return (
        "retry-after" in response.headers
        or response.headers.get("x-ratelimit-remaining") == "0"
        or "rate limit" in body
    )


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=UTC)
                return max(0.0, (retry_time - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError):
                pass
    return min(1.0 * (2**attempt), 8.0)


def _blob_payload(content: FileContent) -> dict[str, str]:
    if isinstance(content, bytes):
        return {
            "content": base64.b64encode(content).decode("ascii"),
            "encoding": "base64",
        }
    return {"content": content, "encoding": "utf-8"}


def _json_payload(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise GhApiError(response.status_code, "GitHub returned invalid JSON.") from exc


def _clean_repo_path(path: str) -> str:
    raw = path.replace("\\", "/")
    if raw.startswith("/") or (len(raw) >= 2 and raw[0].isalpha() and raw[1] == ":"):
        raise ValueError(f"Invalid repository path: {path!r}")
    normalized = raw.strip("/")
    parts = normalized.split("/")
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid repository path: {path!r}")
    return normalized


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise GhApiError(200, f"GitHub response did not contain {key!r}.")
    return value


def _nested_string(payload: dict[str, Any], outer: str, inner: str) -> str:
    nested = payload.get(outer)
    if not isinstance(nested, dict):
        raise GhApiError(200, f"GitHub response did not contain {outer!r}.")
    return _required_string(nested, inner)
