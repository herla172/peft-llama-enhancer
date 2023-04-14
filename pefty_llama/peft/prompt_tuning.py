import torch
import torch.nn as nn
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class AddSoftPrompt(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftC