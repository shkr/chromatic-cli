from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich import print_json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from dotenv import load_dotenv  
load_dotenv()
from .db import list_builds, list_diff_ids, list_projects, bulk_insert_label_embeddings, clear_label_embeddings, count_label_embeddings
from .index import index_diffs
from .write import write_datasets, write_json_record
from .group import group_diffs


app = typer.Typer(help="Chromatic dataset CLI (Python implementation)")


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
    project_id: str = typer.Option(..., "--project"),
    build_id: str = typer.Option(..., "--build"),
    temperature: float = typer.Option(
        0.05,
        "--temperature",
        help="Temperature for label distribution softmax (match trainer).",
    ),
) -> None:
    stats = index_diffs(project_id=project_id, build_id=build_id, temperature=temperature)
    print_json(data=stats)



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


@app.command("list-projects")
def list_projects_cmd() -> None:
    """List all project IDs in the database."""
    projects = list_projects()
    print_json(data={"projects": projects, "count": len(projects)})


@app.command("list-builds")
def list_builds_cmd(
    project_id: str = typer.Option(..., "--project"),
) -> None:
    """List all build IDs for a project."""
    builds = list_builds(project_id)
    print_json(data={"project_id": project_id, "builds": builds, "count": len(builds)})


@app.command("list-diffs")
def list_diffs_cmd(
    project_id: str = typer.Option(..., "--project"),
    build_id: str = typer.Option(..., "--build"),
) -> None:
    """List all diff IDs for a project and build."""
    diffs = list_diff_ids(project_id, build_id)
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


