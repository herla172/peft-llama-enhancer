
import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pefty_llama.modeling import LLaMAModel, NoInitLinear, NoInit8bitLinear, RotaryEmbedding, apply_rotary_pos_emb, check_nan
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class IA3Attention(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads