
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

    def gradient_checkpointing_enable(self):
        self.config.gradient_checkpointing = True

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        self.model.embed_tokens.register_forward_hook(make_inputs_require_grads)


class LLaMAInnerModel(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: peft.PeftConfig):
        super().__init__()
        self.config = config
        self.peft_config = peft_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim, dtype=config.dtype)
        self.layers = nn.ModuleList([
            LLaMALayer(config=config, peft_config=peft_config)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(dim=config.dim)

        if self.peft_config.peft_mode == peft.PEFT_PROMPT:
            self.peft_prompt = peft.AddSoftPrompt(config=config, peft_config=peft_config)

        if self.peft_config.peft_mode == peft.PEFT_LORA and self.peft_config.lora_embedding:
            self.peft_lora_embed = peft.LoRAEmbed(config=config, peft_config=peft_config)

    def forward(self,
                input_ids,
                attention_mask,
                cos, sin,
                kv_cache=None):
        """
        :param input_ids: [batch_size, seq_len]
        :param attention_mask: [batch_size=1, num_heads=1, seq_len, seq_len]
        :param cos: for RoPE
        :param sin: for RoPE
        :param kv_cache: See init_kv_cache.
        """
        hidden_states = self.embed_tokens(input_ids).to(self.config.dtype)
        if self.peft_config.peft_mode == peft.PEFT_LORA and self.peft_config.lora_embedding:
            hidden_states += self.peft_lora_embed(input_ids).to(self.config.dtype)

        if self.peft_config.peft_mode == peft.PEFT_PROMPT:
            if kv_cache is None or kv_cache[0]["key"].shape[2] == 0:
                # Only add prompt if kv_cache is None (full forward pass) or if kv_cache is empty (first decode step)
                hidden_states = self.peft_prompt(hidden_states)

        new_kv_cache = []
        for layer_i, layer in enumerate(self.layers):
            if kv_cache:
                # dict(
                #   key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
                #   value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
                # )
                layer_kv_cache = kv_cache[layer_i]
            else:
                layer_kv_cache = None

            if self.config.gradient_checkpointing:
                layer_out = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    cos, sin,
                    layer_kv_cache,
                )
            else:
                layer_out = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    cos=cos, sin=sin,
                    kv_cache=layer_kv_cache,
                )
            hidden_states, out_layer_kv_cache = layer_out
            if kv_cache:
                new_kv_cache.append(out_layer_kv_cache)
        hidden_states = self.norm(hidden_states)
        output = {
            "hidden_states": hidden_states
        }
        if kv_cache:
            output["kv_cache"] = new_kv_cache
        return output


class LLaMALayer(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: peft.PeftConfig):
        super().__init__()
        self.config = config
        self.peft_config = peft_config
        self.self_attn = Attention(config=config, peft_config=peft_config)
        self.mlp = MLP(config=config, peft_config=peft_config)
        self.input_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)
        self.post_attention_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)

        if self.peft_config.peft_mode == peft.PEFT_ADAPTER:
            if self.peft_config.adapter_version == "houlsby":
                self.peft_adapter_attn = peft.Adapter(config=config, peft_config=peft_config)
            self.peft_adapter_mlp = peft.Adapter(config=config, peft_config=peft_config)

        if self.peft_config.peft_mode == peft.PEFT_BITFIT:
            self.peft_input_layernorm_bias = peft.BitFitAddBias(dim=config.dim, peft_config=peft_config)
            self.peft_post_attention_layernorm_bias = peft.BitFitAddBias(dim=config.dim, peft_config=peft_config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        cos, sin,
        kv_cache=None,
    ):
        # 1) Self-attention
        # [batch_size, seq_len, hidden_dim]
        normed_hidden_states = self.input_layernorm(hidden_states).to(self.config.dtype)
        if self.peft_config.peft_mode == peft.PEFT_BITFIT:
            normed_hidden_states = self.peft_input_layernorm_bias(normed_hidden_states)
        # dict(
        #   attn_output = [batch_size, seq_len, hidden_dim]
        #   kv_cache = dict(
        #     key = [batch_size, num_heads, kv_seq_len, head_dim]
        #     value = [batch_size, num_heads, kv_seq_len, head_dim]
        #   )
        # )
        check_nan(normed_hidden_states)
        raw_self_attn_output = self.self_attn(
            hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            cos=cos, sin=sin,
        )
        # [batch_size, seq_len, hidden_dim]
        attn_out = raw_self_attn_output["attn_output"]
        if self.peft_config.peft_mode == peft.PEFT_ADAPTER \
                and self.peft_config.adapter_version == peft.ADAPTER_VERSION_HOULSBY:
            attn_out = self.peft_adapter_attn(attn_out)

        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states + attn_out
        check_nan(hidden_states)
        # 2) FFN
        # [batch_size, seq_len, hidden_dim]
        post_normed_hidden_states = self.post_attention_layernorm(hidden_states)
        if self.peft_config.peft_mode == peft.PEFT_BITFIT:
            post_normed_hidden_states = self.peft_post_attention_layernorm_bias(post_normed_hidden_states)

        mlp_out = self.mlp(post_normed_hidden_states)
        if self.peft_config.peft_mode == peft.PEFT_ADAPTER:
            mlp_out = self.peft_adapter_mlp(mlp_out)

        hidden_states = hidden_states + mlp_out
        check_nan(hidden_states)
        # if kv_cache:
        #     return {
        #         "hidden_states": hidden_states,
        #         "kv_cache": raw_self_attn_output["kv_cache"],
        #     }
        #
        # return {"hidden_states": hidden_states}
        if kv_cache:
            return hidden_states, raw_self_attn_output["kv_cache"]
        else:
            return hidden_states, None


class MLP(nn.Module):
    def __init__(
        self,
        config: LLaMAConfig,
        peft_config: peft.PeftConfig,
        multiple_of: int = 256,
    ):
        super().__init__()
        self.config = config
        self.peft_config = peft_config
        dim = config.dim
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        if config.use_8bit:
            self.gate_proj = NoInit8bitLinear(dim, hidden_dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.up_proj = NoInit8bitLinear(dim, hidden_dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.down_proj = NoInit8bitLinear(hidden_dim, dim, bias=False, threshold=6.0, has_fp16_weights=False)
        else:
            self.gate_proj = NoInitLinear(dim, hidden_dim, bias=False, dtype=config.dtype)
            self.up_proj = NoInitLinear(dim, hidden_dim, bias=False, dtype=config.dtype)
            self.down_proj = NoInitLinear(hidden_dim, dim, bias=False, dtype=config.dtype)

        if self.peft_config.peft_mode == peft.PEFT_LORA and self.peft_config.lora_mlp:
            self.gate_proj_lora = peft.LoRA(config=config, peft_config=peft_config,
                                            input_dim=dim, output_dim=hidden_dim)
            self.up_proj_lora = peft.LoRA(config=config, peft_config=peft_config,
                                          input_dim=dim, output_dim=hidden_dim)
            self.down_proj_lora = peft.LoRA(config=config, peft_config=peft_config,
                                            input_dim=dim, output_dim=hidden_dim)
        if self.peft_config.peft_mode == peft.PEFT_IA3:
            self.peft_ia3 = peft.IA3ForMLP(config, peft_config=peft_config)
        if self.peft_config.peft_mode == peft.PEFT_BITFIT:
            self.peft_gate_proj_bias = peft.BitFitAddBias(dim=hidden_dim, peft_config=peft_config)
            self.peft_up_proj_bias = peft.BitFitAddBias(dim=hidden_dim, peft_config=peft_config)
            self.peft_down_proj_bias = peft.BitFitAddBias(dim=dim, peft_config=peft_config)

    def forward(self, x):
        gate_proj = self.gate_proj(x)
        up_proj = self.up_proj(x)
        if self.peft_config.peft_mode == peft.PEFT_LORA and self.peft_config.lora_mlp:
            gate_proj += self.gate_proj_lora(x)
            up_proj += self.up_proj_lora(x)
        if self.peft_config.peft_mode == peft.PEFT_BITFIT:
            gate_proj = self.peft_gate_proj_bias(gate_proj)
            up_proj = self.peft_gate_proj_bias(up_proj)

        intermediate_state = F.silu(gate_proj) * up_proj
        if self.peft_config.peft_mode == peft.PEFT_IA3:
            intermediate_state = self.peft_ia3(intermediate_state)

        down_proj = self.down_proj(intermediate_state)
        if self.peft_config.peft_mode == peft.PEFT_LORA and self.peft_config.lora_mlp:
            down_proj = self.down_proj_lora(x)
        if self.peft_config.peft_mode == peft.PEFT_BITFIT:
            down_proj = self.peft_down_proj_bias(down_proj)

        return down_proj


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.float16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)