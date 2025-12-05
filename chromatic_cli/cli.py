from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich import print_json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from dotenv import load_dotenv  
load_dotenv()
from .db import list_diff_ids, bulk_insert_label_embeddings, clear_label_embeddings, count_label_embeddings, get_unindexed_project_build_pairs, count_unindexed_diffs, list_projects_with_limit, list_builds_with_status
from .index import index_diffs
from .write import write_datasets, write_json_record
from .group import group_diffs


app = typer.Typer(help="Chromatic dataset CLI (Python implementation)")


@app.command()
def init() -> None:
    """Initialize: download model weights and index labels."""
    import csv
    from .clip import get_clip_pipeline, encode_text
    from .layout import get_layout_predictor
    
    labels_csv = Path(__file__).parent.parent / "datasets" / "labels.csv"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        # Download and load CLIP model
        clip_task = progress.add_task("[cyan]Downloading CLIP model...", total=None)
        clip_model, _, clip_tokenizer = get_clip_pipeline()
        progress.remove_task(clip_task)
        print("[green]✓ CLIP model ready")
        
        # Download and load Layout model
        layout_task = progress.add_task("[cyan]Downloading Layout model...", total=None)
        get_layout_predictor()
        progress.remove_task(layout_task)
        print("[green]✓ Layout model ready")
        
        # Load and index labels
        with open(labels_csv, 'r') as f:
            reader = csv.DictReader(f)
            labels = [row['text'] for row in reader if row.get('text', '').strip()]
        
        labels = list(set(labels))
        if not labels:
            print_json(data={"error": "No labels found in CSV file"})
            return
        
        # Clear and reindex labels
        clear_label_embeddings()
        
        embed_task = progress.add_task("[green]Indexing styles...", total=len(labels))
        items = []
        for label in labels:
            embedding = encode_text(clip_model, clip_tokenizer, label)
            items.append((label, embedding))
            progress.update(embed_task, advance=1)
        
        bulk_insert_label_embeddings(items)
    
    total = count_label_embeddings()
    print_json(data={
        "status": "initialized",
        "clip_model": "ready",
        "layout_model": "ready", 
        "labels_indexed": len(items),
        "total_labels": total,
    })


@app.command()
def write(
    file: Optional[List[str]] = typer.Option(
        None, "--file", "-f", help="Path(s) or globs to dataset JSON files"
    ),
    json_string: Optional[str] = typer.Option(
        None, "--json", help="Inline JSON record to ingest"
    ),
) -> None:
    if not file and not json_string:
        raise typer.BadParameter("Provide --file or --json")

    stats = {}
    if file:
        stats = write_datasets(file)
    if json_string:
        stats = write_json_record(json_string)
    print_json(data=stats)


@app.command()
def index(
    project_id: Optional[str] = typer.Option(None, "--project", help="Project ID to index. If not provided, indexes all unindexed builds."),
    build_id: Optional[str] = typer.Option(None, "--build", help="Build ID to index. If not provided, indexes all unindexed builds."),
    temperature: float = typer.Option(
        0.05,
        "--temperature",
        help="Temperature for label distribution softmax (match trainer).",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum number of builds to index."),
) -> None:
    """Index diffs to compute embeddings. If project/build not specified, indexes all unindexed builds."""
    # If both project_id and build_id are provided, index that specific pair
    if project_id is not None and build_id is not None:
        stats = index_diffs(project_id=project_id, build_id=build_id, temperature=temperature)
        print_json(data=stats)
        return
    
    # If only one is provided, that's an error
    if project_id is not None or build_id is not None:
        raise typer.BadParameter("Must provide both --project and --build, or neither to index all unindexed builds.")
    
    # Get all unindexed (project_id, build_id) builds
    unindexed_builds = get_unindexed_project_build_pairs()
    
    if not unindexed_builds:
        print_json(data={"message": "No unindexed builds found.", "processed": 0})
        return
    
    # Apply limit to number of builds
    if limit is not None:
        unindexed_builds = unindexed_builds[:limit]
    
    # Count total unindexed diffs across all builds
    total_diffs = sum(count_unindexed_diffs(p, b) for p, b in unindexed_builds)
    
    all_stats = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        # Create a task for overall progress
        overall_task = progress.add_task(
            f"[cyan]Indexing {len(unindexed_builds)} builds ({total_diffs} diffs)...",
            total=len(unindexed_builds),
        )
        
        for proj_id, bld_id in unindexed_builds:
            progress.update(overall_task, description=f"[cyan]Indexing {proj_id}/{bld_id}...")
            stats = index_diffs(project_id=proj_id, build_id=bld_id, temperature=temperature)
            all_stats.append({
                "project_id": proj_id,
                "build_id": bld_id,
                **stats,
            })
            progress.update(overall_task, advance=1)
    
    total_processed = sum(s.get("processed", 0) for s in all_stats)
    print_json(data={
        "message": f"Indexed {len(unindexed_builds)} builds",
        "total_builds": len(unindexed_builds),
        "total_diffs_processed": total_processed,
        "details": all_stats,
    })



@app.command()
def group(
    project_id: str = typer.Option(..., "--project"),
    build_id: str = typer.Option(..., "--build"),
) -> None:
    """
    Generate dendrogram data plus stable groups and neighbor metadata.
    """
    dendrogram_data = group_diffs(project_id=project_id, build_id=build_id)
    print_json(data=dendrogram_data)


@app.command("list")
def list_cmd(
    project_id: Optional[str] = typer.Option(None, "--project", help="List builds for this project"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of results to return"),
) -> None:
    """List projects, or builds for a specific project."""
    if project_id is not None:
        builds = list_builds_with_status(project_id=project_id, limit=limit)
        print_json(data={"project_id": project_id, "builds": builds, "count": len(builds)})
    else:
        projects = list_projects_with_limit(limit=limit)
        print_json(data={"projects": projects, "count": len(projects)})


@app.command("list-diffs")
def list_diffs_cmd(
    project_id: str = typer.Option(..., "--project", help="Project ID"),
    build_id: str = typer.Option(..., "--build", help="Build ID"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of diffs to return"),
) -> None:
    """List diff IDs for a project and build."""
    diffs = list_diff_ids(project_id, build_id)
    diffs = diffs[:limit]
    print_json(data={"project_id": project_id, "build_id": build_id, "diffs": diffs, "count": len(diffs)})


@app.command("index-labels")
def index_labels_cmd(
    csv_path: str = typer.Option(..., "--csv", help="Path to CSV file with 'text' column"),
    clear: bool = typer.Option(False, "--clear", help="Clear existing label embeddings before indexing"),
) -> None:
    """Index text labels from a CSV file by computing CLIP embeddings."""
    import csv
    from .clip import get_clip_pipeline, encode_text
    
    # Load labels from CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        labels = [row['text'] for row in reader if row.get('text', '').strip()]
    
    # Deduplicate labels
    labels = list(set(labels))
    
    if not labels:
        print_json(data={"error": "No labels found in CSV file", "indexed": 0})
        return
    
    # Clear existing embeddings if requested
    if clear:
        clear_label_embeddings()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        # Load CLIP model
        load_task = progress.add_task("[cyan]Loading CLIP model...", total=None)
        clip_model, _, clip_tokenizer = get_clip_pipeline()
        progress.remove_task(load_task)
        
        # Compute embeddings with progress bar
        embed_task = progress.add_task("[green]Encoding labels...", total=len(labels))
        items = []
        for label in labels:
            embedding = encode_text(clip_model, clip_tokenizer, label)
            items.append((label, embedding))
            progress.update(embed_task, advance=1)
        
        # Insert into database
        insert_task = progress.add_task("[yellow]Saving to database...", total=None)
        bulk_insert_label_embeddings(items)
        progress.remove_task(insert_task)
    
    total = count_label_embeddings()
    print_json(data={"indexed": len(items), "total": total, "csv_path": csv_path})


def main() -> None:
    app()


