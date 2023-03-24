
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

        # (batch_size, num_heads, q_seq_len, head_dim)
        query_states = self.q_proj(hidden_states).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos=cos, sin=sin)
        if kv_cache:
            key_states = torch.cat([kv_cache["key"], key_states], dim=2)
            value_states = torch.cat([kv_cache["value"], value_states], dim=2)

        # IA3-specific:
        query_states = query_states * self.peft_l_k
        value_states = value_states * self.peft_l_v
        # end of IA3-specific

        scores = torch.matmul(
            query_states, key_states.transpose(3, 2).type_as(query_states) / math.sqrt(self.head_dim)
        )
        scores += attention_mask

        # (batch_size, num_heads, q_seq_len, kv_seq_len)
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        # (batch_size, num_heads, q_seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value_states.type_as(query_states))
        # (batch_size, q_seq_len, hidden_dim)