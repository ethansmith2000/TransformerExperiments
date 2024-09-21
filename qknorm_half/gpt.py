#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from types import SimpleNamespace

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import time

import pynvml


def normalize(tensor):
    # eps = torch.finfo(tensor.dtype).eps
    eps = 1e-6
    norm = tensor.norm(dim=-1, keepdim=True)
    norm_clamped = torch.where(norm > eps, norm, eps)
    out = tensor / norm_clamped
    return out

class NewGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, 
                        mode="knorm",# ["none", "qknorm", "knorm"]
                    ):
        super().__init__(config, is_cross_attention=False, layer_idx=None)
        self.query_norm = normalize if mode == "qknorm" else nn.Identity()
        self.key_norm = normalize if (mode == "qknorm" or mode == "knorm") else nn.Identity()
        self.scaling = self.head_dim ** -0.5
        self.softmax_temp = None
        if mode == "knorm" or mode == "qknorm":
            self.softmax_temp = nn.Parameter(torch.ones(1, self.num_heads, 1, 1) * 10, requires_grad=True)
        if mode == "qknorm":
            self.scaling = 1.0
        print(mode)
            
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        query = self.query_norm(query)
        key = self.key_norm(key)
        if self.softmax_temp is not None:
            key = key * self.softmax_temp
        query = query * self.scaling

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value_min = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value_min = torch.full([], mask_value_min, dtype=attn_weights.dtype, device=attn_weights.device)
            # mask_value_max = torch.full([], mask_value_max, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value_min)
            # attn_weights_neg = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value_max)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = self.attn_dropout(attn_weights.type(value.dtype))

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


def patch_attn(model, mode="knorm"):
    conf = model.config
    idx = 0

    for n,m in model.named_modules():
        if hasattr(m, "attn"):
            # if idx in indices:
            del m.attn
            m.add_module("attn", NewGPT2Attention(conf, is_cross_attention=False, layer_idx=None, mode=mode))
            # print("activated", idx)
            idx += 1
            # print('current idx', idx)


extra_args = {
    "mode": "knorm",
}


def get_run_name():
    base_str = "base"
    if args["mode"] == "knorm" or args["mode"] == "knorm":
        base_str = args["mode"]
    args["output_dir"] = f"{args['base_output_dir']}/{base_str}"