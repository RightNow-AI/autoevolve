"""Pure HTML parsing for SINTEF VRPTW best known solution tables."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from html.parser import HTMLParser

from campaigns.vrp.objective import BOUND_CLAIM_PREFIX, encode_objective_value

_INSTANCE = re.compile(r"\b((?:RC|C|R)\d+(?:_\d+)*)\b", re.IGNORECASE)
_INTEGER = re.compile(r"^\d+$")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class PageParseResult:
    bounds: tuple[dict[str, str], ...]
    row_errors: tuple[str, ...]


class _TableCollector(HTMLParser):
    """Collect every table row on the page, whatever it is nested inside.

    Nesting depth is deliberately not tracked. SINTEF wraps the best known
    table inside an outer layout table, so a collector that only read
    depth-one tables saw the wrapper and skipped every data row, returning
    zero bounds from a page that holds sixty of them. Rows are gathered into
    one sequence and the header detection below decides which of them mean
    anything.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(_clean("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None

    @property
    def tables(self) -> list[list[list[str]]]:
        """Present the flat row list in the shape the caller already expects."""

        return [self.rows] if self.rows else []


def _clean(value: str) -> str:
    return _WHITESPACE.sub(" ", value.replace("\xa0", " ")).strip()


def _header_columns(row: list[str]) -> dict[str, int] | None:
    normalized = [re.sub(r"[^a-z0-9]+", " ", cell.casefold()).strip() for cell in row]

    def find(*terms: str) -> int | None:
        return next(
            (index for index, cell in enumerate(normalized) if any(term in cell for term in terms)),
            None,
        )

    columns = {
        "instance": find("instance", "problem", "name"),
        "vehicles": find("vehicle"),
        "distance": find("distance"),
        "reference": find("reference", "author", "attribution", "contributor"),
        "date": find("date", "year"),
    }
    if all(index is not None for index in columns.values()):
        return {name: int(index) for name, index in columns.items()}
    return None


def _instance_name(value: str) -> str | None:
    match = _INSTANCE.search(value)
    return match.group(1) if match else None


def _vehicle_count(value: str) -> int:
    cleaned = value.replace(" ", "")
    if not _INTEGER.fullmatch(cleaned):
        raise ValueError(f"vehicle count is not an integer: {value!r}")
    result = int(cleaned)
    if result < 1:
        raise ValueError("vehicle count must be positive")
    return result


def _distance(value: str) -> float:
    cleaned = value.replace("\xa0", "").replace(" ", "")
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    result = float(cleaned)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("distance must be a non-negative finite number")
    return result


def _row_values(row: list[str], columns: dict[str, int] | None) -> tuple[str, ...] | None:
    if columns is not None:
        if max(columns.values()) >= len(row):
            return None
        values = tuple(row[columns[name]] for name in columns)
    else:
        instance_index = next(
            (index for index, cell in enumerate(row) if _instance_name(cell) is not None),
            None,
        )
        if instance_index is None or instance_index + 5 > len(row):
            return None
        values = tuple(row[index] for index in range(instance_index, instance_index + 5))
    if _instance_name(values[0]) is None:
        return None
    return values


def parse_sintef_page(html: str, source_url: str, checked_on: str) -> PageParseResult:
    """Parse every recognizable instance row without inventing missing data."""

    parser = _TableCollector()
    parser.feed(html)
    bounds: list[dict[str, str]] = []
    row_errors: list[str] = []
    seen: set[str] = set()
    for table in parser.tables:
        columns: dict[str, int] | None = None
        for row in table:
            candidate_columns = _header_columns(row)
            if candidate_columns is not None:
                columns = candidate_columns
                continue
            values = _row_values(row, columns)
            if values is None:
                continue
            instance_cell, vehicles_cell, distance_cell, reference, date = values
            instance = _instance_name(instance_cell)
            assert instance is not None
            key = instance.casefold()
            if key in seen:
                continue
            try:
                vehicles = _vehicle_count(vehicles_cell)
                distance = _distance(distance_cell)
                if not reference or not date:
                    raise ValueError("reference and date must both be present")
            except ValueError as exc:
                row_errors.append(f"{instance}: {exc}")
                continue
            bounds.append(
                {
                    "claim": f"{BOUND_CLAIM_PREFIX}{instance}",
                    "value": encode_objective_value(vehicles, distance),
                    "direction": "lexicographic_lower_is_better",
                    "who_and_year": f"{reference}; {date}",
                    "source_url": source_url,
                    "checked_on": checked_on,
                    "how_to_recheck": (
                        f"re-read the {instance} row and confirm {vehicles} vehicles, "
                        f"distance {distance:.2f}, attribution {reference!r}, and date {date!r}"
                    ),
                }
            )
            seen.add(key)
    return PageParseResult(tuple(bounds), tuple(row_errors))
