"""Seed: the classical recursive superpermutation construction."""

# EVOLVE-BLOCK-START
def build(n: int, deadline: float | None = None) -> str:
    """Return the standard recursive superpermutation on symbols 1 through n."""

    del deadline
    result = "1"
    for symbol in range(2, n + 1):
        parts: list[str] = []
        for index, character in enumerate(result):
            parts.append(character)
            if index < symbol - 2:
                continue
            window = result[index - (symbol - 2) : index + 1]
            if len(set(window)) == symbol - 1:
                parts.append(str(symbol))
                parts.append(window)
        result = "".join(parts)
    return result
# EVOLVE-BLOCK-END
