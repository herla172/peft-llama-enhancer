
# based on https://github.com/zphang/minimal-llama/blob/c37e481136f118a16f77f50cdf5e867ed5dafbf9/minimal_llama/pref/llama_simple2.py

import os
import json
import math
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

import bitsandbytes as bnb
import tqdm.auto as tqdm

from accelerate import init_empty_weights
from transformers.utils.bitsandbytes import set_module_8bit_tensor_to_device
from transformers import (
    LlamaConfig as HF_LlamaConfig,
    LlamaForCausalLM as HF_Llama,
)
import pefty_llama.peft as peft
from pefty_llama.configuration import LLaMAConfig, LLAMA_CONFIG_DICT


class LLaMAModel(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: peft.PeftConfig):
        super().__init__()
        self.config = config
        self.peft_config = peft_config
        self.model = LLaMAInnerModel(config=config, peft_config=peft_config)
        self.lm_head = NoInitLinear(config.dim, config.vocab_size, bias=False, dtype=config.dtype)

        if self.peft_config.peft_mode == peft.PEFT_PREFIX:
            self.peft_prefixes = peft.SoftPrefixes(config=config, peft_config=peft_config)
        if self.peft_config.peft_mode == peft.PEFT_LORA and self.peft_config.lora_embedding:
            self.peft_lora_lm_head = peft.LoRA(config=config, peft_config=peft_config,
                                               output_dim=config.vocab_size)

    def forward(self,
                input_ids):
        """Forward pass (with full decode sequence, intended for training or loss-scoring)

        :param input_ids: [batch_size, seq_len]
        :return: logits [batch_size, seq_len]
        """
        # 1) Create masks
        # decoder mask
        # [batch_size, num_heads=1, q_len=seq_len, kv_len=seq_len]
        attention_mask = create_attention_mask(input_ids=input_ids, dtype=self.config.dtype)
        input_ids_for_rope = input_ids
        if self.peft_config.peft_mode == peft.PEFT_PREFIX:
            attention_mask = torch.cat([
                zeros_like([1, 1, input_ids.shape[1], self.peft_config.num_prefix_tokens], tensor=attention_mask),
                attention_mask,
            ], dim=3)

        if self.peft_config.peft_mode in peft.PEFT_PROMPT:
            input_ids_for_rope = torch.cat([
                torch.ones([input_ids.shape[0], self.peft_config.num_prefix_tokens],
                           dtype=input_ids.dtype, device=input_ids.device),
                input_ids,
            ], dim=1)
            # Easier to just remake the attention mask
            attention_mask = create_attention_mask(input_ids=input_ids_for_rope, dtype=self.config.dtype)
        rope_embed_ids = create_rope_embed_ids(input_ids=input_ids_for_rope)
        cos, sin = self.get_cos_sin(rope_embed_ids)

        if self.peft_config.peft_mode == peft.PEFT_PREFIX:
            kv_cache = self.peft_prefixes(batch_size=input_ids.shape[0])
        else:
            kv_cache = None

        # 2) Forward pass
        # [batch_size, seq_len, hidden_dim]
        model_out = self.model(
            input_ids,
            attention_mask=attention_mask,
            cos=cos, sin=sin,
            kv_cache=kv_cache,
        )
        # [batch_size, seq_len, vocab_size]
        logits = self.lm_head(model_out["hidden_states"])
        if self.peft_config.peft_mode == peft.PEFT_LORA and self.peft_config.lora_embedding:
            logits += self.peft_lora_lm_head(model_out["hidden_states"])
        return logits

    def init_kv_cache(self, input_ids):
        # noinspection GrazieInspection
        """Initialize KV cache for decoding.

        A KV cache consists of a list of dicts (one per layer):
            dict(
              key = [batch_size, num_heads, kv_seq_len=0, head_dim]
              value = [batch_size, num_heads, kv_seq_len=0, head_dim]
            )

        :param input_ids: [batch_size, dec_seq_len]
        :return: 0-length kv_cache
        """
        kv_cache = []
        batch_size = input_ids.shape[0]
        num_heads = self.config.n_heads
        head_dim = self.config.head_dim
        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            kv_cache.append({
                "key": torch.zeros([batch_size, num_heads, 0, head_dim]).to(device=device, dtype=self.config.dtype),
                "value": torch.zeros([batch_size, num_heads, 0, head_dim]).to(device=device, dtype=self.config.dtype),
            })
        return kv_cache

    def generate(self, input_ids, generation_length: int = 20,
                 return_output_only=True):
        """Generate tokens with efficient caching of KV.

        TODO: Add stopping conditions
        TODO: Add sampling capabilities

        :param input_ids: [batch_size, enc_seq_len]
        :param generation_length: int
        :param return_output_only:
        :return: [batch_size, generation_length]
        """
        original_input_ids = input_ids
        batch_size, seq_len = input_ids.shape
        # noinspection PyUnresolvedReferences