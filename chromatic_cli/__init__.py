"""
Chromatic Python CLI package.

This module exposes high-level functions that can be imported 
by notebooks or Lambda or CLI functions.
"""
import dotenv

dotenv.load_dotenv()

from .write import write_datasets 
from .index import index_diffs
from .group import group_diffs

__all__ = ["write_datasets", "index_diffs", "group_diffs"]




