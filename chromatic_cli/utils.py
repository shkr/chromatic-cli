from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, List


def compute_diff_id(base_capture_url: str) -> str:
    return hashlib.sha256(base_capture_url.encode("utf-8")).hexdigest()


def iter_dataset_files(paths: Iterable[str]) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            for child in sorted(path.glob("*.json")):
                yield child
        elif any(ch in raw for ch in "*?[]"):
            for match in sorted(Path().glob(raw)):
                if match.is_file():
                    yield match
        else:
            yield path


def load_json_file(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected array at {path}, got {type(data)}")




