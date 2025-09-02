# api/src/registry/__init__.py
from .registry import register, all_names, get, load_from_entry_point
from .types import TrainerSpec

__all__ = ["register", "all_names", "get", "load_from_entry_point", "TrainerSpec"] 
