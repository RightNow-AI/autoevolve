"""Content-addressed candidate storage tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoevolve.core.store import ContentStore, content_ref


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configured = tmp_path / "home"
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(configured))
    return configured


def test_store_round_trip_and_materialize(home: Path, tmp_path: Path) -> None:
    store = ContentStore(home)
    files = {"main.py": "print('ok')\n", "pkg/data.txt": "value\n"}
    ref = store.put(files)
    assert store.get(ref) == files
    destination = tmp_path / "materialized"
    store.materialize(ref, destination)
    assert (destination / "main.py").read_text(encoding="utf-8") == files["main.py"]
    assert (destination / "pkg" / "data.txt").read_text(encoding="utf-8") == "value\n"


def test_put_is_idempotent(home: Path) -> None:
    store = ContentStore(home)
    files = {"candidate.py": "answer = 42\n"}
    first = store.put(files)
    second = store.put(files)
    assert first == second
    assert [path.name for path in store.root.iterdir()] == [first]


def test_ref_is_independent_of_dict_insertion_order(home: Path) -> None:
    first = {"a.py": "a\n", "b.py": "b\n"}
    second = {"b.py": "b\n", "a.py": "a\n"}
    assert content_ref(first) == content_ref(second)
    assert ContentStore(home).put(first) == ContentStore(home).put(second)


@pytest.mark.parametrize(
    "path",
    ["../escape.py", "/absolute.py", "a/../../escape.py", "C:\\absolute.py", "."],
)
def test_store_rejects_paths_outside_candidate_root(home: Path, path: str) -> None:
    with pytest.raises(ValueError, match="relative path"):
        ContentStore(home).put({path: "bad\n"})


def test_store_normalizes_windows_separators(home: Path) -> None:
    store = ContentStore(home)
    ref = store.put({"pkg\\module.py": "value = 1\n"})
    assert store.get(ref) == {"pkg/module.py": "value = 1\n"}


def test_store_rejects_invalid_reference(home: Path) -> None:
    with pytest.raises(ValueError, match="invalid content reference"):
        ContentStore(home).get("../not-a-ref")
