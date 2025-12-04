from __future__ import annotations

import json
from typing import List, Optional, Sequence, Tuple
from .db import bulk_insert_project_build, insert_project_build_row
from .utils import compute_diff_id, iter_dataset_files, load_json_file

REQUIRED_FIELDS = [
    "appId",
    "buildNumber",
    "baseCaptureImageUrl",
    "headCaptureImageUrl",
    "comparisonCaptureImageUrl",
]


def _validate_record(record: dict) -> Tuple[bool, Optional[str]]:
    for field in REQUIRED_FIELDS:
        if field not in record or not record[field]:
            return False, field
    return True, None


def _prepare_row(record: dict, dataset_name: str) -> dict:
    metadata = record.get("metadata") or {}
    metadata = dict(metadata)
    metadata.setdefault("dataset", dataset_name)
    diff_id = compute_diff_id(record["baseCaptureImageUrl"])
    return {
        "project_id": record["appId"],
        "build_id": int(record["buildNumber"]),
        "diff_id": diff_id,
        "component_name": record.get("componentName", "") or "",
        "story_name": record.get("storyName", "") or "",
        "head_capture_image_url": record.get("headCaptureImageUrl", "") or "",
        "head_capture_image_html": record.get("headCaptureImageHtml", "") or "",
        "base_capture_image_url": record.get("baseCaptureImageUrl", "") or "",
        "base_capture_image_html": record.get("baseCaptureImageHtml", "") or "",
        "comparison_capture_image_url": record.get(
            "comparisonCaptureImageUrl", ""
        )
        or "",
        "metadata": metadata,
        "status": record.get("testStatus"),
    }


def write_datasets(paths: Sequence[str]) -> dict:
    inserted = 0
    skipped: List[dict] = []
    batch: List[dict] = []

    for path in iter_dataset_files(paths):
        data = load_json_file(path)
        dataset_name = path.name
        for record in data:
            ok, missing = _validate_record(record)
            if not ok:
                skipped.append({"record": record, "reason": f"missing {missing}"})
                continue
            batch.append(_prepare_row(record, dataset_name))
            if len(batch) >= 500:
                bulk_insert_project_build(batch)
                inserted += len(batch)
                batch.clear()

    if batch:
        bulk_insert_project_build(batch)
        inserted += len(batch)

    return {"inserted": inserted, "skipped": skipped}


def write_json_record(json_string: str, dataset_name: str = "stdin") -> dict:
    record = json.loads(json_string)
    ok, missing = _validate_record(record)
    if not ok:
        raise ValueError(f"Missing required field {missing}")
    row = _prepare_row(record, dataset_name)
    insert_project_build_row(row)
    return {"inserted": 1}




