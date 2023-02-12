
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


@dataclasses.dataclass
class LLaMAConfig:
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int = 32000
    max_seq_length: int = 2048
    dtype = torch.float16
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_8bit: bool = False

    @property
    def head_dim(self):
        return self.dim // self.n_heads


LLAMA_7B_CONFIG = LLaMAConfig(
    dim=4096,
    n_layers=32,
    n_heads=32,
)

LLAMA_CONFIG_DICT = {
    "7b": LLAMA_7B_CONFIG,
}


class LLaMAModel(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.model = LLaMAInnerModel(config)
        self.lm_head = NoInitLinear(config.dim, config.vocab_size, bias=False, dtype=config.dtype)

    @classmethod
    def from_pretrained(cls, model_name_or_path, use_8bit=False):
        """Load model from a huggingface model name or path."""
        hf_config = HF_LlamaConfig.from_pretrained(model_name_or_path)

        config = LLaMAConfig(
            vocab_size=hf_config.vocab_size,
            dim=hf_config.hidden_size,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            max_seq_length=hf_config.max_position_embeddings,
            dtype=hf_config.dtype,
            pad_token_id=hf_config.pad_token_id,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            use_8bit=use_8bit,
        )

        raise NotImplementedError()
        model = cls(config)

        # Load weights from huggingface model to the disk if needed
        if os.path.isdir(model_name_or_path):
            hf_model_path = model_name_or_path
        else:
            hf_model_path = hf_config.cache_dir
            hf_model = HF_LLaMA.from_pretrained(hf_model_path, config=hf_config)
            hf_model.save_pretrained(hf_model_path)

        return model

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
        rope_embed_ids = create_rope_embed_ids(input_ids=input_ids)
        cos, sin = self.get_cos_sin(rope_embed_ids)

        # 2) Forward pass
        # [batch_size, seq_len, hidden_dim]
        model_out = self.model(
            input_ids,
            attention_mask=attention_mask,
            cos=cos, sin=sin,
        )
        # [batch_size, seq_len, vocab_size]
        logits = self.lm_head(model_out["hidden_states"])
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

    def generate(self, input_ids, generation_length: 20):
        """Generate tokens with efficient caching of KV.

        TODO: Add stopping conditions
        TODO: Add sampling capabilities

        :param input_ids: [batch_size, enc_seq_len]
        :param generation_length: int
        :return: [batch_size, generation_length]
        """
        original_input_ids = input_ids
        batch_size, seq_len = input_ids.shape
        # noinspection PyUnresolvedReferences
        num_valid_tokens = (input_ids != self.config.pad_token_id).long().sum(dim=1)

        # 1) Setup
        if input_ids is None:
            # [batch_size, dec_seq_len=1]
            input_ids = torch.LongTensor(
                [[self.config.pad_token_id]] * batch_size
            ).to(self.lm_head.weights.device)
        # See: init_kv_cache. list[dict]
        kv_cache = self.init_kv_cache(input_ids)
        generated_token_ids_list = [original_input_ids]
        total_seq_len = seq_len
