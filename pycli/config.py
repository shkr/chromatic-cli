"""
Configuration utilities shared across CLI commands.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEFAULT_DB = Path(os.getenv("CHROMATIC_DB_PATH", "chromatic.db")).expanduser()


@lru_cache(maxsize=1)
def get_db_path() -> Path:
    return DEFAULT_DB


def get_surya_weights_path() -> Optional[str]:
    return os.getenv("SURYA_WEIGHTS_PATH")


def get_clip_checkpoint_dir() -> Optional[str]:
    return os.getenv("CLIP_CHECKPOINT_DIR")


def get_clip_model_name() -> str:
    return os.getenv("CLIP_MODEL_NAME", "ViT-B-32")


def get_clip_pretrained() -> str:
    return os.getenv("CLIP_PRETRAINED", "metaclip_fullcc")


def get_asset_token() -> Optional[str]:
    return os.getenv("CHROMATIC_ASSET_TOKEN")




