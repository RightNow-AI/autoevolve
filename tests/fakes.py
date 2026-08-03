"""Deterministic test doubles shared across unit tests.

FakeEndpoint duck-types mutate.models.ModelEndpoint: anything with a
.chat(messages, **kw) -> str works. Scripted responses make operator tests
deterministic and offline.
"""

from __future__ import annotations


class FakeEndpoint:
    """Returns scripted responses in order, then repeats the last one."""

    def __init__(self, responses: list[str]):
        assert responses, "FakeEndpoint needs at least one scripted response"
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]
