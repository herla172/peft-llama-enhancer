import torch.nn as nn
import torch.nn.functional as F
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class Adapter(nn.Module):
    def __init__(self, config: LLaMA