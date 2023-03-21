
from typing import Any
from dataclasses import dataclass, field

import torch

PEFT_PREFIX = "prefix"
PEFT_PROMPT = "prompt"
PEFT_ADAPTER = "adapter"
PEFT_PREFIX_ADAPTER = "prefix_adapter"
PEFT_LORA = "lora"
PEFT_IA3 = "ia3"
PEFT_BITFIT = "bitfit"
NO_PEFT = "nothing"

ADAPTER_VERSION_HOULSBY = "houlsby"
ADAPTER_VERSION_PFEIFFER = "pfeiffer"


@dataclass
class PeftConfig:
    peft_mode: str = field()
    peft_dtype: Any = field(default=torch.float32)
