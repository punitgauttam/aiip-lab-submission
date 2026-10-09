#!/usr/bin/env python3
"""Pack genuine cache entries consumed by the Lab 7 gate for offline CI."""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["AIP_OFFLINE"] = "1"
os.environ.setdefault("AIP_TRACE_DIR", tempfile.mkdtemp(prefix="lab7-cache-pack-"))

from aip import cache  # noqa: E402
from aip.config import settings  # noqa: E402
from labs.lab7.gate import measure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "labs/lab7/offline_cache"),
        help="output directory containing calls.sqlite3",
    )
    args = parser.parse_args()
    target_dir = Path(args.output).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "calls.sqlite3"
    if target.exists():
        parser.error(f"refusing to overwrite existing cache: {target}")

    seen_keys: set[str] = set()
    original_get = cache.get

    def record_hits(key: str):
        value = original_get(key)
        if value is not None:
            seen_keys.add(key)
        return value

    cache.get = record_hits
    try:
        metrics = measure()
    finally:
        cache.get = original_get

    source_path = settings.cache_dir / "calls.sqlite3"
    source = sqlite3.connect(source_path)
    try:
        rows = source.execute(
            "SELECT key, kind, request, response, created_at FROM calls "
            "WHERE key IN ({})".format(",".join("?" for _ in seen_keys)),
            tuple(seen_keys),
        ).fetchall()
    finally:
        source.close()

    selected_keys = {row[0] for row in rows}
    missing = seen_keys - selected_keys
    if missing:
        raise RuntimeError(
            f"{len(missing)} consumed cache entries disappeared from {source_path}"
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[1]] = counts.get(row[1], 0) + 1
    if counts.get("embed", 0) < 209 or counts.get("chat", 0) == 0:
        raise RuntimeError(
            f"captured cache is incomplete: {counts}; expected >=209 embeddings "
            "and chat responses for the golden gate"
        )

    with tempfile.NamedTemporaryFile(
        prefix="lab7-offline-", suffix=".sqlite3", dir=target_dir, delete=False
    ) as temp_file:
        temporary_path = Path(temp_file.name)
    try:
        destination = sqlite3.connect(temporary_path)
        try:
            destination.execute(
                """CREATE TABLE calls (
                    key TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    request TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )
            destination.executemany("INSERT INTO calls VALUES (?, ?, ?, ?, ?)", rows)
            destination.commit()
        finally:
            destination.close()
        temporary_path.replace(target)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Packed {len(rows)} genuine cache entries from {source_path}")
    print(f"Cache kinds: {counts}")
    print(f"Gate metrics: {metrics}")
    print(f"Offline gate cache: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
