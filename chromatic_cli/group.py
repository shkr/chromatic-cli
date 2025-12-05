from __future__ import annotations
from .dendrogram import (
    build_combined_dendrogram,
    build_tree_from_data,
    find_stable_groups,
    GROUP_DISTANCE_THRESHOLD,
)

def group_diffs(
    project_id: str,
    build_id: str
) -> dict:
    
     # Generate dendrogram data
    dendro, labels = build_combined_dendrogram(
        project_id=project_id,
        emb_types=["base", "head", "mask_b", "mask_h"],  # All embedding types
        build_id=build_id,
        method="average",
        combination_method="consensus",
        weights={"base": 0.4, "head": 0.4, "mask_b": 0.1, "mask_h": 0.1},  # Weighted combination
        consensus_method="mean"
    )

     # Convert numpy array to list for JSON serialization
    cluster_labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
    
    # Convert nodes to the expected format
    nodes = []
    for node in dendro["nodes"]:
        node_data = {
            "id": node["id"],
            "leaf": node["leaf"]
        }
        if node["leaf"]:
            node_data["label"] = node["label"]
        else:
            node_data["left"] = node["left"]
            node_data["right"] = node["right"]
            node_data["distance"] = node["distance"]
            node_data["count"] = node["count"]
        
        nodes.append(node_data)
    
    # Build tree and find stable groups
    tree_data = {
        "nodes": nodes,
        "root": dendro["root"],
    }
    tree = build_tree_from_data(tree_data)
    raw_stable_groups = find_stable_groups(tree, threshold=GROUP_DISTANCE_THRESHOLD)
    
    # Extract diff IDs from labels (format: "build_id/diff_id")
    stable_groups = [
        [label.split("/")[-1] for label in group if label]
        for group in raw_stable_groups
    ]
        
    return {
        "nodes": nodes,
        "root": dendro["root"],
        "n_leaves": dendro["n_leaves"],
        "linkage": dendro["linkage"],
        "cluster_labels": cluster_labels,
        "stable_groups": stable_groups,
    }

__all__ = [
    "group_diffs",
]




