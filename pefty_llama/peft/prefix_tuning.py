
import torch
import torch.nn as nn
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class SoftPrefixes(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        self.config = config
        self.peft_config = peft_config
        if self.peft_config.prefix_use_mlp:
            if self.peft_config.prefix_mlp_intermediate_size is not None:
                intermediate_size = self.peft_config.prefix_mlp_intermediate_size
            else:
                intermediate_size = self.config.dim

            self.initial = nn.Parameter(
                torch.randn(peft_config.num_prefix_tokens, config.dim, dtype=peft_config.peft_dtype)
            )