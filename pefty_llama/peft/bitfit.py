import torch
import torch.nn as nn
from .configuration import PeftConfig


class BitFitAddBias(nn.Module):
    def __init__(self, dim: int, peft_config: PeftConfig):
        super().__init__()