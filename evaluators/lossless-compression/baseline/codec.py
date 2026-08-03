"""Pure-Python byte-aligned LZ77 seed for the lossless compression pack.

Candidate contract:
compress(data: bytes) returns a self-contained bytes blob.
decompress(blob: bytes) returns the exact original bytes.
"""

from __future__ import annotations


def compress(data: bytes) -> bytes:
    """Encode bytes with literal runs and fixed-window LZ77 backreferences."""
    # Evolution may change only the implementation inside this block.
    # EVOLVE-BLOCK-START
    if type(data) is not bytes:
        raise TypeError("data must be exact bytes")

    window = 4_095
    maximum_match = 130
    maximum_history = 64
    output = bytearray(b"AELZ1")
    output.extend(len(data).to_bytes(8, "big"))
    histories: dict[bytes, list[int]] = {}
    literals = bytearray()

    def flush_literals() -> None:
        if not literals:
            return
        output.append(len(literals) - 1)
        output.extend(literals)
        literals.clear()

    def remember(position: int) -> None:
        if position + 3 > len(data):
            return
        key = data[position : position + 3]
        history = histories.setdefault(key, [])
        history.append(position)
        while history and position - history[0] > window:
            del history[0]
        if len(history) > maximum_history:
            del history[: len(history) - maximum_history]

    index = 0
    while index < len(data):
        best_length = 0
        best_distance = 0
        if index + 3 <= len(data):
            key = data[index : index + 3]
            candidates = histories.get(key, [])
            limit = min(maximum_match, len(data) - index)
            for previous in reversed(candidates):
                distance = index - previous
                if distance > window:
                    break
                length = 3
                while (
                    length < limit
                    and data[previous + length] == data[index + length]
                ):
                    length += 1
                if length > best_length:
                    best_length = length
                    best_distance = distance
                    if length == limit:
                        break

        if best_length >= 3:
            flush_literals()
            output.append(0x80 | (best_length - 3))
            output.extend(best_distance.to_bytes(2, "big"))
            for position in range(index, index + best_length):
                remember(position)
            index += best_length
            continue

        literals.append(data[index])
        remember(index)
        index += 1
        if len(literals) == 128:
            flush_literals()

    flush_literals()
    return bytes(output)
    # EVOLVE-BLOCK-END


def decompress(blob: bytes) -> bytes:
    """Decode the byte-aligned LZ77 stream produced by compress."""
    # Evolution may change only the implementation inside this block.
    # EVOLVE-BLOCK-START
    if type(blob) is not bytes:
        raise TypeError("blob must be exact bytes")
    if len(blob) < 13 or blob[:5] != b"AELZ1":
        raise ValueError("invalid AELZ1 header")

    original_size = int.from_bytes(blob[5:13], "big")
    output = bytearray()
    index = 13
    while len(output) < original_size:
        if index >= len(blob):
            raise ValueError("truncated token stream")
        control = blob[index]
        index += 1
        if control < 0x80:
            length = control + 1
            end = index + length
            if end > len(blob) or len(output) + length > original_size:
                raise ValueError("invalid literal run")
            output.extend(blob[index:end])
            index = end
            continue

        length = (control & 0x7F) + 3
        if index + 2 > len(blob):
            raise ValueError("truncated backreference")
        distance = int.from_bytes(blob[index : index + 2], "big")
        index += 2
        if distance == 0 or distance > len(output):
            raise ValueError("invalid backreference distance")
        if len(output) + length > original_size:
            raise ValueError("backreference exceeds original size")
        for _ in range(length):
            output.append(output[-distance])

    if index != len(blob):
        raise ValueError("trailing bytes after token stream")
    return bytes(output)
    # EVOLVE-BLOCK-END
