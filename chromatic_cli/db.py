from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np

from .config import get_db_path
from .models import VectorDiffItem

Connection = sqlite3.Connection

_CONN: Optional[Connection] = None


def _init_connection(path: Path) -> Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _ensure_schema(conn)
    return conn


def get_connection() -> Connection:
    global _CONN
    if _CONN is None:
        db_path = get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _CONN = _init_connection(db_path)
    return _CONN


@contextmanager
def scoped_cursor() -> Iterator[sqlite3.Cursor]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def _ensure_schema(conn: Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS project_build (
            project_id TEXT NOT NULL,
            build_id TEXT NOT NULL,
            diff_id TEXT NOT NULL,
            component_name TEXT DEFAULT '',
            story_name TEXT DEFAULT '',
            head_capture_image_url TEXT DEFAULT '',
            head_capture_image_html TEXT DEFAULT '',
            base_capture_image_url TEXT DEFAULT '',
            base_capture_image_html TEXT DEFAULT '',
            comparison_capture_image_url TEXT DEFAULT '',
            metadata TEXT,
            status TEXT,
            base_embedding TEXT,
            head_embedding TEXT,
            mask_b_embedding TEXT,
            mask_h_embedding TEXT,
            PRIMARY KEY (project_id, build_id, diff_id)
        );

        CREATE TABLE IF NOT EXISTS vector_diff_item (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            build_id TEXT NOT NULL,
            diff_id TEXT NOT NULL,
            item_index INTEGER NOT NULL,
            label TEXT NOT NULL,
            position REAL,
            confidence REAL,
            image_src TEXT,
            image TEXT,
            bbox TEXT,
            polygon TEXT,
            embedding TEXT,
            texts TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (project_id, build_id, diff_id, item_index)
        );

        CREATE INDEX IF NOT EXISTS idx_project_build_lookup
            ON project_build(project_id, build_id, diff_id);

        CREATE INDEX IF NOT EXISTS idx_vector_diff_lookup
            ON vector_diff_item(project_id, build_id, diff_id, item_index);

        CREATE TABLE IF NOT EXISTS label_embedding (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT UNIQUE NOT NULL,
            embedding TEXT NOT NULL
        );
        """
    )


def insert_project_build_row(row: dict) -> None:
    payload = dict(row)
    payload.setdefault("component_name", "")
    payload.setdefault("story_name", "")
    payload["metadata"] = json.dumps(payload.get("metadata") or {})
    # Handle embedding columns - serialize lists to JSON if provided
    for emb_col in ("base_embedding", "head_embedding", "mask_b_embedding", "mask_h_embedding"):
        val = payload.get(emb_col)
        if val is not None and not isinstance(val, str):
            payload[emb_col] = json.dumps(val)
        else:
            payload.setdefault(emb_col, None)
    with scoped_cursor() as cursor:
        cursor.execute(
            """
            INSERT OR REPLACE INTO project_build (
                project_id, build_id, diff_id,
                component_name, story_name,
                head_capture_image_url, head_capture_image_html,
                base_capture_image_url, base_capture_image_html,
                comparison_capture_image_url, metadata, status,
                base_embedding, head_embedding, mask_b_embedding, mask_h_embedding
            )
            VALUES (
                :project_id, :build_id, :diff_id,
                :component_name, :story_name,
                :head_capture_image_url, :head_capture_image_html,
                :base_capture_image_url, :base_capture_image_html,
                :comparison_capture_image_url, :metadata, :status,
                :base_embedding, :head_embedding, :mask_b_embedding, :mask_h_embedding
            )
            """,
            payload,
        )


def _serialize_embedding(val) -> Optional[str]:
    """Serialize an embedding value to JSON string if needed."""
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return json.dumps(val)


def bulk_insert_project_build(rows: Iterable[dict]) -> None:
    with scoped_cursor() as cursor:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO project_build (
                project_id, build_id, diff_id,
                component_name, story_name,
                head_capture_image_url, head_capture_image_html,
                base_capture_image_url, base_capture_image_html,
                comparison_capture_image_url, metadata, status,
                base_embedding, head_embedding, mask_b_embedding, mask_h_embedding
            ) VALUES (
                :project_id, :build_id, :diff_id,
                :component_name, :story_name,
                :head_capture_image_url, :head_capture_image_html,
                :base_capture_image_url, :base_capture_image_html,
                :comparison_capture_image_url, :metadata, :status,
                :base_embedding, :head_embedding, :mask_b_embedding, :mask_h_embedding
            )
            """,
            [
                {
                    **row,
                    "metadata": json.dumps(row.get("metadata") or {}),
                    "component_name": row.get("component_name", ""),
                    "story_name": row.get("story_name", ""),
                    "base_embedding": _serialize_embedding(row.get("base_embedding")),
                    "head_embedding": _serialize_embedding(row.get("head_embedding")),
                    "mask_b_embedding": _serialize_embedding(row.get("mask_b_embedding")),
                    "mask_h_embedding": _serialize_embedding(row.get("mask_h_embedding")),
                }
                for row in rows
            ],
        )


def fetch_diff_record(project_id: str, build_id: str, diff_id: str) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.execute(
        """
        SELECT * FROM project_build
        WHERE project_id = ? AND build_id = ? AND diff_id = ?
        """,
        (project_id, build_id, diff_id),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    data = dict(row)
    if data.get("metadata"):
        try:
            data["metadata"] = json.loads(data["metadata"])
        except json.JSONDecodeError:
            data["metadata"] = {"raw": data["metadata"]}
    else:
        data["metadata"] = {}
    return data


def list_projects() -> List[str]:
    """Return all unique project_ids in the database."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT DISTINCT project_id FROM project_build ORDER BY project_id"
    )
    return [row["project_id"] for row in cursor.fetchall()]


def list_builds(project_id: str) -> List[str]:
    """Return all unique build_ids for a given project."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT DISTINCT build_id FROM project_build WHERE project_id = ? ORDER BY build_id",
        (project_id,),
    )
    return [row["build_id"] for row in cursor.fetchall()]


def list_diff_ids(project_id: str, build_id: str) -> List[str]:
    """Return all diff_ids for a given project and build."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT diff_id FROM project_build WHERE project_id = ? AND build_id = ? ORDER BY diff_id",
        (project_id, build_id),
    )
    return [row["diff_id"] for row in cursor.fetchall()]


def list_diff(
    project_id: str,
    build_id: str,
) -> List[dict]:
    conn = get_connection()
    params: List = [project_id, build_id]
    query = """
        SELECT * FROM project_build
        WHERE project_id = ? AND build_id = ?
    """
    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    result = []
    for row in rows:
        data = dict(row)
        metadata = data.get("metadata")
        if metadata:
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"raw": metadata}
        else:
            metadata = {}
        data["metadata"] = metadata
        result.append(data)
    return result


def update_project_build_embeddings(
    project_id: str,
    build_id: str,
    diff_id: str,
    base_embedding: Optional[List[float]] = None,
    head_embedding: Optional[List[float]] = None,
    mask_b_embedding: Optional[List[float]] = None,
    mask_h_embedding: Optional[List[float]] = None,
) -> None:
    """Update embedding columns for an existing project_build row."""
    with scoped_cursor() as cursor:
        cursor.execute(
            """
            UPDATE project_build
            SET base_embedding = ?,
                head_embedding = ?,
                mask_b_embedding = ?,
                mask_h_embedding = ?
            WHERE project_id = ? AND build_id = ? AND diff_id = ?
            """,
            (
                json.dumps(base_embedding) if base_embedding is not None else None,
                json.dumps(head_embedding) if head_embedding is not None else None,
                json.dumps(mask_b_embedding) if mask_b_embedding is not None else None,
                json.dumps(mask_h_embedding) if mask_h_embedding is not None else None,
                project_id,
                build_id,
                diff_id,
            ),
        )
        if cursor.rowcount == 0:
            print(
                f"[WARN] insert_project_build_embedding updated 0 rows for "
                f"{project_id} {build_id} {diff_id}"
            )


def save_vector_diff_items(
    project_id: str,
    build_id: str,
    diff_id: str,
    items: Sequence[VectorDiffItem],
) -> None:
    with scoped_cursor() as cursor:
        cursor.execute(
            "DELETE FROM vector_diff_item WHERE project_id=? AND build_id=? AND diff_id=?",
            (project_id, build_id, diff_id),
        )
        rows = []
        for index, item in enumerate(items):
            rows.append(
                (
                    project_id,
                    build_id,
                    diff_id,
                    index,
                    item.label,
                    item.position,
                    item.confidence,
                    item.image_src,
                    item.image,
                    json.dumps(item.bbox) if item.bbox is not None else None,
                    json.dumps(item.polygon) if item.polygon is not None else None,
                    json.dumps(item.embedding),
                    json.dumps(item.texts or []),
                )
            )
        cursor.executemany(
            """
            INSERT INTO vector_diff_item (
                project_id, build_id, diff_id, item_index,
                label, position, confidence, image_src, image,
                bbox, polygon, embedding, texts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def get_vector_diff_items(
    project_id: str,
    build_id: int,
    diff_id: str,
) -> List[Dict[str, Any]]:
    """Retrieve vector_diff_item rows for a given diff from SQLite."""
    conn = get_connection()
    cursor = conn.execute(
        """
        SELECT
            item_index,
            label,
            position,
            confidence,
            image_src,
            image,
            bbox,
            polygon,
            embedding,
            texts
        FROM vector_diff_item
        WHERE project_id = ? AND build_id = ? AND diff_id = ?
        ORDER BY item_index
        """,
        (project_id, build_id, diff_id),
    )
    rows = cursor.fetchall()
    result: List[Dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "item_index": row["item_index"],
                "label": row["label"],
                "position": row["position"],
                "confidence": row["confidence"],
                "image_src": row["image_src"],
                "image": row["image"],
                "bbox": json.loads(row["bbox"]) if row["bbox"] else None,
                "polygon": json.loads(row["polygon"]) if row["polygon"] else None,
                "embedding": json.loads(row["embedding"]) if row["embedding"] else None,
                "texts": json.loads(row["texts"]) if row["texts"] else None,
            }
        )
    return result


def insert_label_embedding(text: str, embedding: List[float]) -> None:
    """Insert a single label embedding into the database."""
    with scoped_cursor() as cursor:
        cursor.execute(
            """
            INSERT OR REPLACE INTO label_embedding (text, embedding)
            VALUES (?, ?)
            """,
            (text, json.dumps(embedding)),
        )


def bulk_insert_label_embeddings(items: List[Tuple[str, List[float]]]) -> None:
    """Bulk insert label embeddings into the database."""
    with scoped_cursor() as cursor:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO label_embedding (text, embedding)
            VALUES (?, ?)
            """,
            [(text, json.dumps(embedding)) for text, embedding in items],
        )


def get_label_embeddings() -> Tuple[np.ndarray, List[str]]:
    """
    Query SQLite to get label embeddings.
    
    Returns:
        Tuple of (label_embeddings, label_texts)
        - label_embeddings: Lx512 matrix of normalized embeddings
        - label_texts: List of L label text strings
    """
    conn = get_connection()
    cursor = conn.execute("SELECT text, embedding FROM label_embedding")
    results = cursor.fetchall()
    
    if not results:
        return np.array([]), []
    
    # Extract embeddings and texts
    label_texts = [row["text"] for row in results]
    label_embeddings = np.array([json.loads(row["embedding"]) for row in results])
    
    # L2-normalize embeddings for cosine similarity
    norms = np.linalg.norm(label_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    label_embeddings = label_embeddings / norms
    
    return label_embeddings, label_texts

def clear_label_embeddings() -> None:
    """Clear all label embeddings from the database."""
    with scoped_cursor() as cursor:
        cursor.execute("DELETE FROM label_embedding")


def count_label_embeddings() -> int:
    """Count the number of label embeddings in the database."""
    conn = get_connection()
    cursor = conn.execute("SELECT COUNT(*) as count FROM label_embedding")
    row = cursor.fetchone()
    return row["count"] if row else 0


def get_unindexed_project_build_pairs() -> List[Tuple[str, str]]:
    """
    Get all (project_id, build_id) pairs that have rows in project_build
    but haven't been indexed yet (base_embedding is NULL).
    
    Returns:
        List of (project_id, build_id) tuples that need indexing.
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        SELECT DISTINCT project_id, build_id 
        FROM project_build 
        WHERE base_embedding IS NULL
        ORDER BY project_id, build_id
        """
    )
    return [(row["project_id"], row["build_id"]) for row in cursor.fetchall()]


def list_projects_with_limit(limit: int = 10) -> List[str]:
    """Get distinct project IDs.
    
    Args:
        limit: Maximum number of projects to return
        
    Returns:
        List of project IDs
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        SELECT DISTINCT project_id 
        FROM project_build 
        ORDER BY project_id
        LIMIT ?
        """,
        (limit,),
    )
    return [row["project_id"] for row in cursor.fetchall()]


def list_builds_with_status(project_id: str, limit: int = 10) -> List[dict]:
    """Get builds for a project with their index status.
    
    Args:
        project_id: Project ID to filter by
        limit: Maximum number of builds to return
        
    Returns:
        List of dicts with build_id and indexed (bool)
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        SELECT build_id, 
               CASE WHEN base_embedding IS NOT NULL THEN 1 ELSE 0 END as indexed
        FROM project_build 
        WHERE project_id = ?
        GROUP BY build_id
        ORDER BY build_id DESC
        LIMIT ?
        """,
        (project_id, limit),
    )
    return [
        {
            "build_id": row["build_id"],
            "indexed": bool(row["indexed"]),
        }
        for row in cursor.fetchall()
    ]


def count_unindexed_diffs(project_id: str, build_id: str) -> int:
    """Count unindexed diffs for a specific project/build pair."""
    conn = get_connection()
    cursor = conn.execute(
        """
        SELECT COUNT(*) as count 
        FROM project_build 
        WHERE project_id = ? AND build_id = ? AND base_embedding IS NULL
        """,
        (project_id, build_id),
    )
    row = cursor.fetchone()
    return row["count"] if row else 0


def insert_project_build_embedding(
    project_id: str,
    build_id: int,
    diff_id: str,
    base_embedding,
    head_embedding,
    mask_b_embedding,
    mask_h_embedding,
) -> None:
    """Update embedding columns for an existing project_build row.
    
    Accepts numpy arrays or lists as embedding inputs.
    """
    # Convert numpy arrays to lists if needed
    def to_list(arr):
        if arr is None:
            return None
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        return arr
    
    with scoped_cursor() as cursor:
        cursor.execute(
            """
            UPDATE project_build
            SET base_embedding = ?,
                head_embedding = ?,
                mask_b_embedding = ?,
                mask_h_embedding = ?
            WHERE project_id = ? AND build_id = ? AND diff_id = ?
            """,
            (
                json.dumps(to_list(base_embedding)) if base_embedding is not None else None,
                json.dumps(to_list(head_embedding)) if head_embedding is not None else None,
                json.dumps(to_list(mask_b_embedding)) if mask_b_embedding is not None else None,
                json.dumps(to_list(mask_h_embedding)) if mask_h_embedding is not None else None,
                project_id,
                build_id,
                diff_id,
            ),
        )

