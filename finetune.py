
import argparse
import os
import math
from dataclasses import dataclass, field
import tqdm.auto as tqdm
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import datasets
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from pefty_llama.peft import PeftConfig
from pefty_llama.modeling_peft import create_model, set_peft_requires_grad


@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    hf_path: str = field()
    model_name: str = field(default="7b")
    use_8bit: bool = field(default=False)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def only_tunable_params(model):
    requires_grad = {k: v.requires_grad for k, v in model.named_parameters()}
    return {
        k: v
        for k, v in model.state_dict().items()
        if k in requires_grad and requires_grad[k]
    }


class ModifiedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        batch_size = inputs["input_ids"].shape[0]

        labels = inputs["input_ids"]
        input_ids = torch.cat([
            torch.ones(batch_size, 1).long().to(labels.device),
            inputs["input_ids"][:, :-1],
        ], dim=1)

        # logits will be 1 block shorter than input_ids, since we're dropping off the first block
        logits = model(input_ids=input_ids)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(
            -1, logits.size(-1)), labels.reshape(-1)
        )