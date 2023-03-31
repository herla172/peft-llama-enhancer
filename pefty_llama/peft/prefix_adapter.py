import torch
import torch.nn as nn
import torch.nn.functional as F
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class PrefixAdapter(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        self.config = config
        self.peft_config = peft_config
        # "batch_size"=1, num_heads, num_prefix_tokens, head_dim
        self.prefix_k = nn.Para