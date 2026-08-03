"""Generate deterministic corpus files for the lossless compression pack."""

from __future__ import annotations

import random
import string
from pathlib import Path

SEED = 1_618_033
NATURAL_SIZE = 8_320
REPETITIVE_SIZE = 8_000
RANDOM_SIZE = 4_096
CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
FIXTURE_NAMES = (
    "natural-language.txt",
    "repetitive.txt",
    "near-random.txt",
)

WORDS = (
    "the",
    "small",
    "engine",
    "measures",
    "each",
    "candidate",
    "against",
    "one",
    "fixed",
    "contract",
    "while",
    "workers",
    "search",
    "through",
    "clear",
    "changes",
    "and",
    "record",
    "every",
    "result",
    "because",
    "evidence",
    "must",
    "remain",
    "honest",
    "when",
    "language",
    "contains",
    "repeated",
    "patterns",
    "a",
    "compressor",
    "can",
    "replace",
    "long",
    "phrases",
    "with",
    "short",
    "references",
    "but",
    "random",
    "symbols",
    "offer",
    "little",
    "structure",
    "for",
    "any",
    "method",
    "to",
    "reuse",
    "this",
    "corpus",
    "moves",
    "from",
    "common",
    "words",
    "toward",
    "related",
    "ideas",
    "so",
    "sentences",
    "sound",
    "regular",
    "without",
    "being",
    "copied",
)


def build_natural_language() -> bytes:
    """Return a seeded word-level Markov sample with stable ASCII bytes."""
    rng = random.Random(SEED)
    output = bytearray()
    current = rng.randrange(len(WORDS))
    sentence_length = rng.randint(7, 15)
    sentence_position = 0

    while len(output) < NATURAL_SIZE:
        word = WORDS[current]
        if sentence_position == 0:
            word = word.capitalize()
        ending = ".\n" if sentence_position + 1 == sentence_length else " "
        output.extend((word + ending).encode("ascii"))

        offsets = (1, 2, 5, 8, 13)
        offset = offsets[rng.randrange(len(offsets))]
        current = (current + offset + sentence_position % 3) % len(WORDS)
        sentence_position += 1
        if sentence_position == sentence_length:
            sentence_position = 0
            sentence_length = rng.randint(7, 15)

    return bytes(output[:NATURAL_SIZE])


def build_repetitive() -> bytes:
    """Return a fixed repeated passage with short and long repetitions."""
    passage = (
        b"measure the candidate, verify the bytes, record the result; "
        b"measure the candidate, verify the bytes, record the result.\n"
    )
    repeated = passage * (REPETITIVE_SIZE // len(passage) + 1)
    return repeated[:REPETITIVE_SIZE]


def build_near_random() -> bytes:
    """Return seeded near-random printable text with no intentional phrases."""
    rng = random.Random(SEED ^ 0xA5A5_A5A5)
    alphabet = (string.ascii_letters + string.digits + string.punctuation + " \n").encode()
    return bytes(alphabet[rng.randrange(len(alphabet))] for _ in range(RANDOM_SIZE))


def build_fixtures() -> dict[str, bytes]:
    """Return every named corpus sample."""
    return {
        "natural-language.txt": build_natural_language(),
        "repetitive.txt": build_repetitive(),
        "near-random.txt": build_near_random(),
    }


def write_fixtures(output_dir: Path = CORPUS_DIR) -> None:
    """Rewrite the corpus with byte-stable generated content."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in build_fixtures().items():
        (output_dir / name).write_bytes(content)


if __name__ == "__main__":
    write_fixtures()
