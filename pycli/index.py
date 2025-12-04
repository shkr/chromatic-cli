from __future__ import annotations

import asyncio
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import normalize

from .layout import download_images, process_images_to_vectors_from_diff_data
from .db import (
    get_label_embeddings,
    insert_project_build_embedding,
    list_diff,
    save_vector_diff_items,
)
from .models import VectorDiffItem

# Mapping from item position to embedding column name
EMBEDDING_POSITION_MAP = {
    0: "base_embedding",      # BaseComponent
    1: "head_embedding",      # HeadComponent
    2: "mask_b_embedding",    # MaskedBaseComponent
    3: "mask_h_embedding",    # MaskedHeadComponent
}


@dataclass
class Instance:
    id: str
    base: np.ndarray      # (m_i, d)
    head: np.ndarray      # (n_i, d)
    mask_b: np.ndarray    # (d,)
    mask_h: np.ndarray    # (d,)


def convert_to_items(items_data: List[dict]) -> List[VectorDiffItem]:
    converted: List[VectorDiffItem] = []
    for item in items_data:
        converted.append(
            VectorDiffItem(
                label=item.get("label", ""),
                position=item.get("position", 0),
                confidence=item.get("confidence", 0),
                image_src=item.get("image_src", ""),
                image=item.get("image", ""),
                bbox=item.get("bbox"),
                polygon=item.get("polygon"),
                embedding=item.get("embedding", []),
                texts=item.get("texts"),
            )
        )
    return converted



def l2norm(A):
    """L2-normalize for cosine similarity."""
    return normalize(A, norm="l2", copy=False)


def softmax_temp(scores, T=0.05):
    """
    Temperature-scaled softmax.
    Lower T => sharper topics; tune 0.03–0.2
    """
    z = scores / max(T, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def compute_label_distribution(vectors: np.ndarray, label_embeddings: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """
    Compute label distribution for a set of vectors.
    
    Args:
        vectors: WxD matrix of vectors (W vectors, D dimensions)
        label_embeddings: LxD matrix of label embeddings (L labels, D dimensions)
        temperature: Temperature for softmax scaling
        
    Returns:
        L-dimensional vector representing the average label distribution
    """
    if vectors.size == 0:
        return np.zeros(label_embeddings.shape[0])
    
    # Normalize input vectors
    vectors_norm = l2norm(vectors)
    
    # Compute cosine similarities: WxL matrix
    similarities = np.dot(vectors_norm, label_embeddings.T)
    
    # Apply temperature-scaled softmax to get probabilities
    label_probs = softmax_temp(similarities, temperature)
    
    # Average across all vectors in the set
    avg_label_dist = np.mean(label_probs, axis=0)
    
    return avg_label_dist

    


def prepare_distribution_inputs(
    items: List[dict],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]]:
    """Aggregate vector diff items into base/head/mask tensors plus metadata."""
    base_embeddings = []
    head_embeddings = []
    base_mask_embeddings = []
    head_mask_embeddings = []

    for item in items:
        embedding = item.get("embedding", [])
        if not embedding:
            continue
        vector = np.array(embedding)
        if item.get("image_src") == "BaseImage" and item.get("label") == "BaseComponent":
            base_embeddings.append(vector)
        elif item.get("image_src") == "HeadImage" and item.get("label") == "HeadComponent":
            head_embeddings.append(vector)
        elif item.get("image_src") == "MaskedBaseImage" and item.get("label") == "MaskedBaseComponent":
            base_mask_embeddings.append(vector)
        elif item.get("image_src") == "MaskedHeadImage" and item.get("label") == "MaskedHeadComponent":
            head_mask_embeddings.append(vector)

    if not base_embeddings or not head_embeddings:
        return None

    base_array = np.array(base_embeddings)
    head_array = np.array(head_embeddings)

    mask_b = (
        base_mask_embeddings[0]
        if base_mask_embeddings
        else base_embeddings[0]
    )
    mask_h = (
        head_mask_embeddings[0]
        if head_mask_embeddings
        else head_embeddings[0]
    )

    meta = {
        "base_count": str(len(base_embeddings)),
        "head_count": str(len(head_embeddings)),
        "mask_b_source": "masked" if base_mask_embeddings else "base",
        "mask_h_source": "masked" if head_mask_embeddings else "head",
    }

    return base_array, head_array, mask_b, mask_h, meta


def serialize_distribution_inputs(
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]
) -> Dict[str, object]:
    base_array, head_array, mask_b, mask_h, meta = arrays
    return {
        "base_vectors": base_array.tolist(),
        "head_vectors": head_array.tolist(),
        "mask_b_vector": mask_b.tolist(),
        "mask_h_vector": mask_h.tolist(),
        "meta": meta,
    }


async def index_single(
    row: dict,
    label_embeddings: np.ndarray,
    label_texts: List[str],
    temperature: float = 0.05,
) -> int:
    """
    Process a single diff row: compute vector diff items, persist them,
    and write label-distribution embeddings that match trainer/index.py.
    """

    base_image, head_image, comparison_image = await download_images(row)
    items = await process_images_to_vectors_from_diff_data(
        base_image,
        head_image,
        comparison_image,
        label_embeddings,
        label_texts,
    )
    save_vector_diff_items(
        project_id=row["project_id"],
        build_id=row["build_id"],
        diff_id=row["diff_id"],
        items=convert_to_items(items),
    )

    prepared = prepare_distribution_inputs(items)
    if prepared is None:
        # Missing required vectors; skip indexing for this diff.
        return 0

    base_array, head_array, mask_b, mask_h, _meta = prepared

    base_dist = compute_label_distribution(base_array, label_embeddings, temperature)
    head_dist = compute_label_distribution(head_array, label_embeddings, temperature)

    mask_b_dist = (
        compute_label_distribution(mask_b.reshape(1, -1), label_embeddings, temperature)
        if mask_b.size > 0
        else np.zeros(label_embeddings.shape[0])
    )
    mask_h_dist = (
        compute_label_distribution(mask_h.reshape(1, -1), label_embeddings, temperature)
        if mask_h.size > 0
        else np.zeros(label_embeddings.shape[0])
    )

    if os.getenv("PYCLI_DEBUG_DIST"):
        debug_payload = {
            "project_id": row["project_id"],
            "build_id": row["build_id"],
            "diff_id": row["diff_id"],
            "base_dist": base_dist.tolist(),
            "head_dist": head_dist.tolist(),
            "mask_b_dist": mask_b_dist.tolist(),
            "mask_h_dist": mask_h_dist.tolist(),
        }
        debug_dir = Path(os.getenv("PYCLI_DEBUG_DIR", "pycli/output"))
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_file = debug_dir / f"dist_{row['diff_id']}.json"
        debug_file.write_text(json.dumps(debug_payload))

    print(f"Indexing {row['project_id']} {row['build_id']} {row['diff_id']}")
    insert_project_build_embedding(
        project_id=row["project_id"],
        build_id=row["build_id"],
        diff_id=row["diff_id"],
        base_embedding=base_dist,
        head_embedding=head_dist,
        mask_b_embedding=mask_b_dist,
        mask_h_embedding=mask_h_dist,
    )
    return len(items)
    


def index_diffs(
    project_id: str,
    build_id: str,
    temperature: float = 0.05,
) -> dict:
    rows = list_diff(project_id, build_id)
    print(f"Number of diffs for project_id: {project_id}, build_id: {build_id}: {len(rows)}")
    if not rows:
        return {
            "processed": 0,
            "diffs": [],
            "error": f"No diffs found for project '{project_id}' build {build_id}. "
                     "Run 'chromatic-cli write' first to load the data.",
        }

    async def _runner():
        processed = []
        label_embeddings, label_texts = get_label_embeddings()
        for row in rows:
            count = await index_single(
                row,
                label_embeddings,
                label_texts,
                temperature=temperature,
            )
            processed.append({"diff_id": row["diff_id"], "items": count})
        return processed

    processed = asyncio.run(_runner())
    return {"processed": len(processed), "diffs": processed}




