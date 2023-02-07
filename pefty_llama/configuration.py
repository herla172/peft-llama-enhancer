
from typing import Any
import dataclasses
import torch


@dataclasses.dataclass
class LLaMAConfig:
    dim: int