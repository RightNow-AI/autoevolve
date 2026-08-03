"""Invalid certificate hidden behind a mapping with changing reads."""

from __future__ import annotations


class HostileCertificate(dict[str, object]):
    """Pretend to be valid on one red_diffs read, then expose all blue."""

    def __init__(self) -> None:
        super().__init__(form="circulant", n=4, red_diffs=[])
        self.reads = 0

    def __getitem__(self, key: str) -> object:
        if key == "red_diffs":
            self.reads += 1
            return [1] if self.reads == 1 else []
        return super().__getitem__(key)


def construct(s: int, n_cap: int, deadline: float | None = None) -> dict[str, object]:
    del deadline
    if s != 3 or n_cap < 4:
        raise ValueError("hostile fixture is only defined for k3-smoke")
    return HostileCertificate()
