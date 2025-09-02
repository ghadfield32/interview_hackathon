# api/src/registry/registry.py
from __future__ import annotations
from importlib import import_module
from typing import Dict, Iterable, Type
from .types import TrainerSpec
from ..trainers.base import BaseTrainer

_REGISTRY: Dict[str, TrainerSpec] = {}

def register(spec: TrainerSpec) -> None:
    _REGISTRY[spec.name] = spec

def all_names() -> Iterable[str]:
    return _REGISTRY.keys()

def get(name: str) -> TrainerSpec:
    return _REGISTRY[name]

def load_from_entry_point(dotted: str, name: str | None = None):
    """
    Load 'pkg.module:ClassName' into registry.
    """
    mod_path, cls_name = dotted.split(":")
    mod = import_module(mod_path)
    cls: Type[BaseTrainer] = getattr(mod, cls_name)
    inst_name = name or getattr(cls, "name", cls_name.lower())
    spec = TrainerSpec(name=inst_name, cls=cls, default_params=cls().default_hyperparams())
    register(spec)
    return spec 
