import torch.nn as nn
import torch.nn.functional as F
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class Adapter(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        self.config = config
        self.peft_config = peft_config
        self.down_proj = nn.Linear(
            config.dim, peft_config.adapter_hidden