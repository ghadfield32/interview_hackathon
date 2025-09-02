# api/src/registry/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Type, Dict, Any
from ..trainers.base import BaseTrainer

@dataclass
class TrainerSpec:
    name: str
    cls: Type[BaseTrainer]
    default_params: Dict[str, Any] 
