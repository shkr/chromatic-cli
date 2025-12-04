from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize

from .db import get_connection

DiffDataWithMeta = Dict[str, Any]

GROUP_DISTANCE_THRESHOLD = 1e-2

EMB_COL_BY_TYPE: Dict[str, str] = {
    "base": "base_embedding",
    "head": "head_embedding",
    "mask_b": "mask_b_embedding",
    "mask_h": "mask_h_embedding",
}
TABLE = "project_build"


@dataclass
class TreeNode:
    id: int | str
    leaf: bool
    label: Optional[str] = None
    distance: Optional[float] = None
    children: List["TreeNode"] = field(default_factory=list)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_diff_id(label: Optional[str]) -> Optional[str]:
    if not label or not isinstance(label, str):
        return None
    parts = label.split("/")
    return parts[-1] if parts else None


def build_tree_from_data(dendrogram_data: Optional[dict]) -> Optional[TreeNode]:
    if not dendrogram_data:
        return None

    nodes_data = dendrogram_data.get("nodes") or []
    if not nodes_data:
        return None

    node_map: MutableMapping[int | str, TreeNode] = {}
    for node in nodes_data:
        node_id = node.get("id")
        if node_id is None:
            continue
        tree_node = TreeNode(
            id=node_id,
            leaf=bool(node.get("leaf")),
            label=node.get("label"),
            distance=_to_float(node.get("distance")),
        )
        node_map[node_id] = tree_node

    for node in nodes_data:
        if node.get("leaf"):
            continue
        tree_node = node_map.get(node.get("id"))
        if tree_node is None:
            continue
        children: List[TreeNode] = []
        for child_id in (node.get("left"), node.get("right")):
            if child_id is None:
                continue
            child_node = node_map.get(child_id)
            if child_node:
                children.append(child_node)
        tree_node.children = children

    root_id = dendrogram_data.get("root")
    return node_map.get(root_id)


def find_stable_groups(
    tree: Optional[TreeNode],
    threshold: float = GROUP_DISTANCE_THRESHOLD,
) -> List[List[str]]:
    if tree is None:
        return []

    groups: List[List[str]] = []

    def collect_leaf_labels(node: Optional[TreeNode], bucket: List[str]) -> None:
        if node is None:
            return
        if node.leaf:
            label = node.label if isinstance(node.label, str) else None
            if label:
                bucket.append(label)
            return
        for child in node.children:
            collect_leaf_labels(child, bucket)

    def traverse(node: Optional[TreeNode]) -> None:
        if node is None:
            return
        if (
            not node.leaf
            and node.children
            and node.distance is not None
            and node.distance < threshold
        ):
            labels: List[str] = []
            for child in node.children:
                collect_leaf_labels(child, labels)
            if labels:
                groups.append(labels)
        for child in node.children:
            traverse(child)

    traverse(tree)
    return groups


def create_stable_groups(
    diffs: Sequence[DiffDataWithMeta],
    tree_data: Optional[dict],
    threshold: float = GROUP_DISTANCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = [
        {
            "id": "all",
            "title": "All",
            "diffs": list(diffs),
        }
    ]

    tree = build_tree_from_data(tree_data)
    if tree is None:
        return groups

    stable_groups = find_stable_groups(tree, threshold=threshold)
    if not stable_groups:
        return groups

    diff_by_id = {diff.get("id"): diff for diff in diffs if diff.get("id")}

    for index, group_labels in enumerate(stable_groups):
        diff_ids = [
            diff_id
            for label in group_labels
            if (diff_id := _extract_diff_id(label)) is not None
        ]
        if not diff_ids:
            continue
        group_diffs = [
            diff_by_id[diff_id] for diff_id in diff_ids if diff_id in diff_by_id
        ]
        if not group_diffs:
            continue
        groups.append(
            {
                "id": f"group-{index}",
                "title": f"Group {index + 1} ({len(group_diffs)} items)",
                "diffs": group_diffs,
            }
        )

    return groups


def build_first_neighbor_map(
    tree_data: Optional[dict],
    diffs: Sequence[DiffDataWithMeta],
    threshold: float = GROUP_DISTANCE_THRESHOLD,
) -> Dict[str, List[DiffDataWithMeta]]:
    if not tree_data or not tree_data.get("nodes") or not diffs:
        return {}

    nodes = tree_data.get("nodes", [])
    nodes_by_id: Dict[int | str, dict] = {}
    parent_by_child: Dict[int | str, dict] = {}
    node_id_by_diff_id: Dict[str, int | str] = {}
    diff_by_id = {diff.get("id"): diff for diff in diffs if diff.get("id")}

    for node in nodes:
        node_id = node.get("id")
        if node_id is None:
            continue
        nodes_by_id[node_id] = node
        if node.get("leaf"):
            diff_id = _extract_diff_id(node.get("label"))
            if diff_id:
                node_id_by_diff_id[diff_id] = node_id
        left = node.get("left")
        right = node.get("right")
        if left is not None:
            parent_by_child[left] = node
        if right is not None:
            parent_by_child[right] = node

    leaf_cache: Dict[int | str, List[str]] = {}

    def collect_leaf_diff_ids(node_id: int | str) -> List[str]:
        if node_id in leaf_cache:
            return leaf_cache[node_id]
        node = nodes_by_id.get(node_id)
        if not node:
            leaf_cache[node_id] = []
            return []
        leaves: List[str] = []
        if node.get("leaf"):
            diff_id = _extract_diff_id(node.get("label"))
            if diff_id:
                leaves.append(diff_id)
        else:
            left = node.get("left")
            right = node.get("right")
            if left is not None:
                leaves.extend(collect_leaf_diff_ids(left))
            if right is not None:
                leaves.extend(collect_leaf_diff_ids(right))
        unique_leaves = list(dict.fromkeys(leaves))
        leaf_cache[node_id] = unique_leaves
        return unique_leaves

    neighbor_map: Dict[str, List[DiffDataWithMeta]] = {}

    for diff_id, diff in diff_by_id.items():
        node_id = node_id_by_diff_id.get(diff_id)
        if node_id is None:
            neighbor_map[diff_id] = []
            continue

        parent_node = parent_by_child.get(node_id)
        if not parent_node:
            neighbor_map[diff_id] = []
            continue

        parent_distance = _to_float(parent_node.get("distance"))
        if parent_distance is None or parent_distance >= threshold:
            neighbor_map[diff_id] = []
            continue

        sibling_leaf_ids = [
            leaf_id for leaf_id in collect_leaf_diff_ids(parent_node.get("id")) if leaf_id != diff_id
        ]
        if not sibling_leaf_ids:
            neighbor_map[diff_id] = []
            continue

        neighbor_diffs = [
            diff_by_id[sibling_id] for sibling_id in sibling_leaf_ids if sibling_id in diff_by_id
        ]
        neighbor_map[diff_id] = neighbor_diffs

    # Ensure that every diff id is present even if no neighbors were found above.
    for diff in diffs:
        diff_id = diff.get("id")
        if diff_id and diff_id not in neighbor_map:
            neighbor_map[diff_id] = []

    return neighbor_map


# ---------- Embedding + dendrogram construction ----------
def load_embeddings(
    project_id: str,
    emb_type: str,
    build_id: str,
) -> Tuple[List[str], np.ndarray]:
    if emb_type not in EMB_COL_BY_TYPE:
        raise ValueError(f"emb_type must be one of {list(EMB_COL_BY_TYPE)}")
    col = EMB_COL_BY_TYPE[emb_type]

    sql = f"""
        SELECT build_id, diff_id, {col} AS embedding
        FROM {TABLE}
        WHERE project_id = ? AND build_id = ?
        ORDER BY build_id, diff_id
    """

    ids: List[str] = []
    embedding_rows: List[np.ndarray] = []

    conn = get_connection()
    cursor = conn.execute(sql, (project_id, build_id))
    for row in cursor.fetchall():
        embedding = row["embedding"]
        if embedding is None:
            continue
        diff_id = row["diff_id"]
        ids.append(f"{build_id}/{diff_id}")
        embedding_rows.append(np.asarray(json.loads(embedding), dtype=np.float32))

    if not embedding_rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, np.vstack(embedding_rows)



# ---------- HIERARCHICAL CLUSTERING ----------
def compute_linkage_cosine(X: np.ndarray,
                           method: str = "average",
                           optimal_ordering: bool = True):
    """
    L2-normalize, then cosine distance via pdist, run SciPy linkage.
    Returns: linkage matrix Z of shape (N-1, 4).
    """
    if X.shape[0] <= 1:
        return None
    Xn = normalize(X, norm="l2", copy=True)
    # pdist with metric='cosine' returns cosine distance = 1 - cosine_similarity
    dists = pdist(Xn, metric="cosine")
    Z = linkage(dists, method=method, optimal_ordering=optimal_ordering)
    return Z


# ---------- DENDROGRAM JSON ----------
def linkage_to_dendrogram(Z: np.ndarray, ids: list[str]):
    """
    Convert SciPy linkage to a compact, plot-agnostic dendrogram structure.

    Returns:
      {
        "nodes": [
           {"id": 0, "leaf": True,  "label": labels[0]},
           ... (N leaves)
           {"id": N, "leaf": False, "left": <int>, "right": <int>, "distance": <float>, "count": <int>},
           ... (N-1 internal merges)
        ],
        "root": <int>,          # id of root node (N + (N-1) - 1)
        "n_leaves": <int>,
        "linkage": Z_as_list    # optional full linkage for auditing
      }
    Node ids: 0..N-1 are leaves in original order; internal nodes are N..(2N-2).
    """
    N = len(ids)
    nodes = [{"id": i, "leaf": True, "label": ids[i]} for i in range(N)]
    if Z is None:
        # Only one leaf or no leaves
        return {
            "nodes": nodes,
            "root": 0 if N == 1 else -1,
            "n_leaves": N,
            "linkage": [] if N == 1 else []
        }

    next_id = N
    for i in range(Z.shape[0]):
        left = int(Z[i, 0])
        right = int(Z[i, 1])
        dist = float(Z[i, 2])
        count = int(Z[i, 3])
        nodes.append({
            "id": next_id,
            "leaf": False,
            "left": left,
            "right": right,
            "distance": dist,
            "count": count
        })
        next_id += 1

    root = nodes[-1]["id"]
    return {"nodes": nodes, "root": root, "n_leaves": N, "linkage": Z.tolist()}


# ---------- OPTIONAL: FLAT CUT + NO-SPLIT GATE ----------
def flat_partition_from_linkage(Z: np.ndarray,
                                max_distance: float | None = None,
                                min_improvement: float = 0.02):
    """
    If max_distance is provided, returns flat labels from fcluster.
    If not, we implement a conservative 'no-split' gate:
      - If second-last merge distance improves cohesion by < min_improvement over single cluster,
        we return one group.
    Returns: labels (np.array of shape (N,), ints starting at 1), or all-ones for 'no-split'.
    """
    if Z is None:
        return np.array([1], dtype=int)

    # A simple conservative rule: cut at a small distance if a big jump happens late.
    if max_distance is not None:
        return fcluster(Z, t=max_distance, criterion='distance')

    # Heuristic: if the last jump (root) isn't much larger than average of previous,
    # don't split.
    d = Z[:, 2]
    if len(d) < 2:
        return np.ones(Z.shape[0] + 1, dtype=int)

    last = d[-1]
    prev_mean = d[:-1].mean()
    gain = last - prev_mean
    if gain < min_improvement:
        return np.ones(Z.shape[0] + 1, dtype=int)
    # Else cut at (prev_mean + small epsilon)
    t = prev_mean + 1e-6
    return fcluster(Z, t=t, criterion='distance')


# ---------- ORCHESTRATOR (PER TYPE) ----------
def build_dendrogram_for_type(project_id: str,
                              emb_type: str,
                              build_id: str | None = None,
                              limit: int | None = None,
                              method: str = "average"):
    ids, X = load_embeddings(project_id, emb_type, build_id=build_id)
    if len(ids) == 0:
        return {"nodes": [], "root": -1, "n_leaves": 0, "linkage": []}, np.array([])

    Z = compute_linkage_cosine(X, method=method, optimal_ordering=True)
    dendro = linkage_to_dendrogram(Z, ids=ids)

    # Optional: get a conservative partition (can be all-ones = 'no split')
    labels = flat_partition_from_linkage(Z, max_distance=None, min_improvement=0.02)

    return dendro, labels


# ---------- MULTI-TYPE COMBINATION METHODS ----------
def compute_consensus_distance_matrix(embeddings_dict: dict[str, np.ndarray], 
                                    weights: dict[str, float] | None = None,
                                    method: str = "mean") -> np.ndarray:
    """
    Compute a consensus distance matrix from multiple embedding types.
    
    Args:
        embeddings_dict: Dict mapping embedding type to embedding matrix
        weights: Optional weights for each embedding type (default: equal weights)
        method: Consensus method - "mean", "median", or "max"
    
    Returns:
        Consensus distance matrix
    """
    if not embeddings_dict:
        raise ValueError("No embeddings provided")
    
    # Set default weights if not provided
    if weights is None:
        weights = {k: 1.0 for k in embeddings_dict.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Compute distance matrices for each embedding type
    distance_matrices = {}
    for emb_type, X in embeddings_dict.items():
        if X.shape[0] <= 1:
            continue
        Xn = normalize(X, norm="l2", copy=True)
        dists = pdist(Xn, metric="cosine")
        distance_matrices[emb_type] = dists
    
    if not distance_matrices:
        raise ValueError("No valid distance matrices computed")
    
    # Combine distance matrices
    if method == "mean":
        consensus_dists = np.average(
            [dists for dists in distance_matrices.values()], 
            axis=0, 
            weights=[weights[emb_type] for emb_type in distance_matrices.keys()]
        )
    elif method == "median":
        consensus_dists = np.median(list(distance_matrices.values()), axis=0)
    elif method == "max":
        consensus_dists = np.maximum.reduce(list(distance_matrices.values()))
    else:
        raise ValueError(f"Unknown consensus method: {method}")
    
    return consensus_dists


def compute_weighted_concatenation(embeddings_dict: dict[str, np.ndarray],
                                 weights: dict[str, float] | None = None) -> np.ndarray:
    """
    Concatenate multiple embedding types with optional weighting.
    
    Args:
        embeddings_dict: Dict mapping embedding type to embedding matrix
        weights: Optional weights for each embedding type
    
    Returns:
        Concatenated and weighted embedding matrix
    """
    if not embeddings_dict:
        raise ValueError("No embeddings provided")
    
    # Set default weights if not provided
    if weights is None:
        weights = {k: 1.0 for k in embeddings_dict.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Weight and concatenate embeddings
    weighted_embeddings = []
    for emb_type, X in embeddings_dict.items():
        if X.shape[0] == 0:
            continue
        weighted_X = X * weights[emb_type]
        weighted_embeddings.append(weighted_X)
    
    if not weighted_embeddings:
        raise ValueError("No valid embeddings to concatenate")
    
    return np.hstack(weighted_embeddings)


def build_combined_dendrogram(project_id: str,
                            emb_types: list[str] | None = None,
                            build_id: str | None = None,
                            method: str = "average",
                            combination_method: str = "consensus",
                            weights: dict[str, float] | None = None,
                            consensus_method: str = "mean") -> tuple[dict, np.ndarray]:
    """
    Build a dendrogram combining multiple embedding types.
    
    Args:
        project_id: Project identifier
        emb_types: List of embedding types to combine (default: all available)
        build_id: Optional build ID filter
        limit: Optional limit on number of samples
        method: Linkage method for hierarchical clustering
        combination_method: How to combine embeddings - "consensus", "concatenation", or "weighted_concatenation"
        weights: Optional weights for each embedding type
        consensus_method: Method for consensus matrix ("mean", "median", "max")
    
    Returns:
        Tuple of (dendrogram_dict, cluster_labels)
    """
    if emb_types is None:
        emb_types = list(EMB_COL_BY_TYPE.keys())
    
    # Load embeddings for all types
    all_embeddings = {}
    common_ids = None
    
    for emb_type in emb_types:
        ids, X = load_embeddings(project_id, emb_type, build_id=build_id)
        if len(ids) == 0:
            continue
        
        # Ensure all embedding types have the same IDs
        if common_ids is None:
            common_ids = ids
        else:
            # Find common IDs across all embedding types
            common_ids = [id for id in common_ids if id in ids]
        
        all_embeddings[emb_type] = X
    
    if not all_embeddings:
        return {"nodes": [], "root": -1, "n_leaves": 0, "linkage": []}, np.array([])
    
    # Filter embeddings to common IDs
    if common_ids:
        for emb_type in all_embeddings:
            ids, X = load_embeddings(project_id, emb_type, build_id=build_id)
            id_to_idx = {id: i for i, id in enumerate(ids)}
            common_indices = [id_to_idx[id] for id in common_ids if id in id_to_idx]
            all_embeddings[emb_type] = X[common_indices]
    
    if len(common_ids) <= 1:
        return {"nodes": [], "root": -1, "n_leaves": 0, "linkage": []}, np.array([])
    
    # Combine embeddings based on method
    if combination_method == "consensus":
        consensus_dists = compute_consensus_distance_matrix(
            all_embeddings, weights=weights, method=consensus_method
        )
        Z = linkage(consensus_dists, method=method, optimal_ordering=True)
    elif combination_method in ["concatenation", "weighted_concatenation"]:
        combined_X = compute_weighted_concatenation(all_embeddings, weights=weights)
        Z = compute_linkage_cosine(combined_X, method=method, optimal_ordering=True)
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")
    
    # Build dendrogram
    dendro = linkage_to_dendrogram(Z, ids=common_ids)
    labels = flat_partition_from_linkage(Z, max_distance=None, min_improvement=0.02)
    
    return dendro, labels


__all__ = [
    "GROUP_DISTANCE_THRESHOLD",
    "DiffDataWithMeta",
    "TreeNode",
    "build_tree_from_data",
    "find_stable_groups",
    "create_stable_groups",
    "build_first_neighbor_map",
    "build_combined_dendrogram",
]

