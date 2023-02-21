
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
        num_valid_tokens = (input_ids != self.config.pad_token_id).long().sum(dim=1)

        # 1) Setup
        if input_ids is None:
            # [batch_size, dec_seq_len=1]
            input_ids = torch.LongTensor(
                [[self.config.pad_token_id]] * batch_size
            ).to(self.lm_head.weights.device)
        # See: init_kv_cache. list[dict]
        if self.peft_config.peft_mode == peft.PEFT_PREFIX:
            kv_cache = self.peft_prefixes(batch_size=input_ids.shape[0])
            num_valid_kv_cache = num_valid_tokens + self.peft_config.num_prefix_tokens
        else:
            kv_cache = self.init_kv_cache(input_ids)
            num_valid_kv_cache = num_valid_tokens
        generated_token_ids_list = [original_input_ids]
        total_seq_len = seq_len

        # 2) First encoding
        # [batch_size=1, num_heads=1, q_len=1, kv_len=1]
        attention_mask = create_attention_mask(input_ids=input_ids, dtype=self.config.dtype)
        input_ids_for_rope = input_ids
        # dict(
        #   hidden_states = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
        #   kv_cache = list[dict(
        #     key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
        #     value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
        #   )]
        # )
        if self.peft_config.peft_mode in (peft.PEFT_PREFIX, peft.PEFT_PROMPT):
            num_prefix_tokens = self.peft_config.num_prefix_tokens
            total_seq_len += num_prefix_tokens
            # [batch_size, num_heads=1, q_len=seq_len, kv_len=num_prefix_tokens + dec_seq_len]
            attention_mask = torch.cat([
                zeros_like([1, 1, input_ids.shape[1], num_prefix_tokens], tensor=attention_mask),
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
        model_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cos=cos, sin=sin,
            kv_cache=kv_cache,
        )
        logits = self.lm_head(model_out["hidden_states"])
        kv_cache = model_out["kv_cache"]
        generated_token_ids = logits.argmax(-1)[
            torch.arange(batch_size, dtype=torch.long, device=input_ids.device),
            num_valid_tokens-1,
        ][:, None]
        generated_token_ids_list.append(generated_token_ids)
        input_ids = generated_token_ids

        # 3) Subsequent steps
        for decode_step in range(generation_length-1):
            num_valid_tokens += 1
            total_seq_len += 1
            # [batch_size=1, num_heads=1, q_len=1, kv_len=1]
            attention_mask = convert_mask_to_soft_mask(create_generation_attention_mask(
                batch_size=batch_size,
                seq_len=total_seq_len,
                num_valid_tokens=num_valid_tokens,
                device=input_ids.device,
            ), dtype=self.config.dtype)
            # dict(
            #   hidden_states = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
            #   kv_cache = list[dict(
            #     key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
            #     value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
            #   )]
            # )
            rope_embed_ids = create_rope_embed_ids(input_ids=input_ids) + num_valid_tokens[:, None]
            cos, sin = self.get_cos_sin(rope_embed_ids)
            model_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                cos=cos, sin=sin,
            )
            # [batch_size, dec_seq_len=1, vocab_size]
            logits = self.lm_head(model_out["hidden_states"])
            kv_cache = model_out["kv_cache"]
            # [batch_size, dec_seq_len=1]
            generated_token_ids = logits.argmax(-1)[:, -1:]
            generated_token_ids_list.append(generated_token_ids)
            input_ids = generated_token_ids
        output = torch.cat(generated_token_ids_list, dim=1)
        if return_output_only:
            output = output[:, seq_len:]
        return output

    def get_cos_sin(self, rope_embed_ids):
        cos = F.embedding(
            rope_embed_ids,
            self.model.layers[0].self_attn.rotary_emb.cos_cached[0, 0].to(rope_embed_ids.device)
        ).to(self.config.dtype)
        sin = F.embedding(
            rope_embed_ids,
            self.model.layers[0].self_attn.rotary_emb.sin_cached[0, 0].to(rope_embed_ids.device)
        ).to(self.config.dtype)
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]
        return cos, sin