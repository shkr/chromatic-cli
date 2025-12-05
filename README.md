## Installation

```bash
cd pycli
pip install -e .
```

## Environment Variables

| Variable | Description |
| --- | --- |
| `CHROMATIC_DB_PATH` | Path to the local SQLite database (default: `./chromatic.db`) |
| `CACHE_DIR` | Directory for model weights (required) |
| `CHROMATIC_ASSET_TOKEN` | Cookie used when downloading Chromatic assets |
| `S3_BUCKET` | S3 bucket for private model downloads |
| `S3_REGION` | S3 region |
| `S3_ACCESS_KEY_ID` | S3 access key |
| `S3_SECRET_ACCESS_KEY` | S3 secret key |

## Usage

### Initialize (First Run)

Download model weights and index labels:

```bash
chromatic-cli init
```

### Ingest Dataset

```bash
chromatic-cli write --file datasets/dataset1.json
```

```python
from chromatic_cli import write_datasets
write_datasets(["datasets/dataset1.json"])
```

### List Project/Build Pairs

```bash
# List 10 pairs with index status
chromatic-cli list

# List 100 pairs
chromatic-cli list --limit 100

# List builds for a specific project
chromatic-cli list --project 59c59bd0183bd100364e1d57
```

```python
from chromatic_cli.db import get_project_build_pairs_with_status
pairs = get_project_build_pairs_with_status(limit=10)
pairs = get_project_build_pairs_with_status(project_id="59c59bd0183bd100364e1d57", limit=100)
```

### List Diffs

```bash
chromatic-cli list-diffs --project 59c59bd0183bd100364e1d57 --build 46973 --limit 20
```

```python
from chromatic_cli.db import list_diff_ids
diffs = list_diff_ids("59c59bd0183bd100364e1d57", "46973")
```

### Index Labels

```bash
chromatic-cli index-labels --csv labels.csv
chromatic-cli index-labels --csv labels.csv --clear
```

```python
from chromatic_cli.db import bulk_insert_label_embeddings, clear_label_embeddings
from chromatic_cli.clip import get_clip_pipeline, encode_text

clear_label_embeddings()
clip_model, _, clip_tokenizer = get_clip_pipeline()
items = [(label, encode_text(clip_model, clip_tokenizer, label)) for label in labels]
bulk_insert_label_embeddings(items)
```

### Index Diffs

```bash
# Index a specific build
chromatic-cli index --project 59c59bd0183bd100364e1d57 --build 46973

# Index all unindexed builds
chromatic-cli index

# Index up to 5 unindexed builds
chromatic-cli index --limit 5
```

```python
from chromatic_cli import index_diffs
index_diffs(project_id="59c59bd0183bd100364e1d57", build_id="46973")
```

### Group Diffs

```bash
chromatic-cli group --project 59c59bd0183bd100364e1d57 --build 46973
```

```python
from chromatic_cli import group_diffs
groups = group_diffs(project_id="59c59bd0183bd100364e1d57", build_id="46973")
```
