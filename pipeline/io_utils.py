import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Iterator, Optional


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_to_jsonable(rec), ensure_ascii=False) + "\n")


def append_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_to_jsonable(rec), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
    import csv

    rows = list(rows)
    if not rows:
        ensure_parent_dir(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        return

    if fieldnames is None:
        # Stable-ish ordering: take keys from first row, then add any extras
        keys = list(rows[0].keys())
        extras = sorted({k for r in rows for k in r.keys()} - set(keys))
        fieldnames = keys + extras

    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
