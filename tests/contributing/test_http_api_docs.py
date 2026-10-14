"""The grid tables in docs/reference/http-api.rst stay aligned: the docs site drops a short row's cell and reports nothing."""

from __future__ import annotations

from pathlib import Path

HTTP_API = Path(__file__).parents[2] / "docs" / "reference" / "http-api.rst"


def grid_tables(lines: list[str]) -> list[tuple[int, list[str]]]:
    """Each run of consecutive grid table lines (the ones starting with + or |) with its first line number."""
    tables: list[tuple[int, list[str]]] = []
    block: list[str] = []
    start = 0
    for number, line in enumerate(lines, 1):
        if line.startswith(("+", "|")):
            if not block:
                start = number
            block.append(line)
        elif block:
            tables.append((start, block))
            block = []
    if block:
        tables.append((start, block))
    return tables


def test_every_grid_table_line_has_its_bars_in_the_same_columns() -> None:
    tables = grid_tables(HTTP_API.read_text(encoding="utf-8").splitlines())
    assert any("/reef/harness/releases/{step}/page" in line for _, block in tables for line in block)
    for start, block in tables:
        columns = [index for index, char in enumerate(block[0]) if char == "+"]
        assert len(columns) >= 2, f"docs/reference/http-api.rst:{start} does not open a grid table"
        for offset, line in enumerate(block):
            where = f"docs/reference/http-api.rst:{start + offset}"
            assert len(line) == len(block[0]), f"{where} is {len(line)} characters wide; the table is {len(block[0])}"
            assert all(line[column] in "+|" for column in columns), f"{where} has a bar off the table's columns"
