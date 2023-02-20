
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