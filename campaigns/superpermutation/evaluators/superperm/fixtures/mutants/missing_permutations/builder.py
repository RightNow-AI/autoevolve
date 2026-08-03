"""Invalid mutant: it returns only one permutation and omits the other ones."""


def build(n: int, deadline: float | None = None) -> str:
    del deadline
    return "123456789"[:n]
