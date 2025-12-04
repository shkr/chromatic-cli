"""
Chromatic Python CLI package.

This module exposes high-level helpers that can be imported by notebooks or Lambda
functions without pulling in the Typer CLI surface.
"""

from .write import write_datasets 
from .index import index_diffs
from .group import group_diffs

__all__ = ["write_datasets", "index_diffs", "group_diffs"]




