
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
        self.head_dim = config.dim // config.n_heads

        if config.use_8bit:
            self.q_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.k_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.v_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.o_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
        else:
            self.q_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
            self.k_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
            self.v_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
            self.o_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

        # IA3-specific parameters:
        self.peft_l_k = nn.Parameter(torch.ones(1, self.n_heads, 1, self.head_dim, dtype=config.dtype))
        self.peft_l_v = nn.Parameter(torch.ones(1, self.n_heads, 1, self.head_dim, dtype=config.dtype))

    def forward(self, hidden_states, attention_mask, cos, sin, kv_cache=None):
        """
        precomputed_kv_hidden_states is for init (pre-compute KV activations, e.g. for added prefixes)
        kv_cache is for generation (cached past KV)
        """
        batch_size, q_seq_len, hidden_dim = hidden_states.size()