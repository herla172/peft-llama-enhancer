import torch
import torch.nn as nn
from .configuration import PeftConfig


class BitFitAddBias(nn.Module):
    def __init__(self, dim: int, peft_config: PeftConfig):
        super().__init__()
        self.peft_config = peft_config
        self.bias = nn.Parameter(torch.zeros(dim, dtype=peft_config.peft_dtype))

    def forward(self, 