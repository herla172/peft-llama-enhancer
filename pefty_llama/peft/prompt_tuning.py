import torch
import torch.nn as nn
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class AddSoftPrompt(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        self.peft_config = peft_config
        self.soft_prompt = nn.Parameter(
            torch.randn(peft_config.num_prefix_tokens, config.dim, dtype=peft_config.peft_dtype)
        )

   