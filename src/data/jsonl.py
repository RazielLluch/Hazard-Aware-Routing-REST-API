import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def find_record(
    path: Path,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    for record in stream_jsonl(path):
        if predicate(record):
            return record
    return None


def count_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
