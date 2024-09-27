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
import psutil



def run_name():
    if args["use_new_attn"]:
        base_str = f"new_attn"#_{args['offset']}"
        if args["neg_version"]:
            base_str += "_neg"
        if args["include_o"]:
            base_str += "_o"
        if args["alternate"]:
            base_str += "_alt"

        args["output_dir"] = f"{args['base_output_dir']}/{base_str}"
    else:
        base_str = "base"
        args["output_dir"] = f"{args['base_output_dir']}/base"


    unique_activations = list(set(args['activations']))
    non_gelu = [a for a in unique_activations if a != "gelu"]
    if len(non_gelu) > 0:
        non_gelu = non_gelu[0]
        indices = tuple([i+1 for i, a in enumerate(args['activations']) if a == non_gelu])
        base_str = base_str + "_{}-{}".format(non_gelu, indices)

def normalize(tensor):
    # eps = torch.finfo(tensor.dtype).eps
    eps = 1e-6
    norm = tensor.norm(dim=-1, keepdim=True)
    norm_clamped = torch.where(norm > eps, norm, eps)
    out = tensor / norm_clamped
    return out



def profile_gpus():
    # print("*"*100)
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f">>> GPU Profiling | GPU: {i} | Used Memory: {info.used / 1024 / 1024} MB")
    pynvml.nvmlShutdown()
    # print("*"*100)

def profile_cpu():
    memory_info = psutil.virtual_memory()
    return memory_info


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class Activation(nn.Module):

    def __init__(self, activation_type: str = 'relu'):
        super().__init__()
        self.activation_type = activation_type
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'gelu':
            self.activation = nn.GELU()
        elif activation_type == 'silu':
            self.activation = nn.SiLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'sin':
            self.activation = lambda x: torch.sin(x) + (x/2)

    def forward(self, x):
        return self.activation(x)


class GPT2MLP(nn.Module):
    def __init__(self, config, activation_type='gelu'):
        super().__init__()
        embed_dim = config.hidden_size
        intermediate_size = embed_dim * 4
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = Activation(activation_type=activation_type)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class NewGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, offset=5, neg_version=False, include_o=False, normalized=False, learnable_softmax=False, inner_offset=0):
        super().__init__(config, is_cross_attention=False, layer_idx=None)
        self.neg_version = neg_version
        self.offset = offset
        self.normalized_attention = normalized
        if learnable_softmax:
            self.softmax_temp = nn.Parameter(torch.ones(1,self.heads,1,1) * 10)
        else:
            self.softmax_temp = None
        
        if not self.neg_version:
            self.c_attn = Conv1D(4 * self.embed_dim, self.embed_dim)
        if include_o:
            self.c_proj_2 = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_proj_2 = None

        self.inner_offset = inner_offset

        self.pos_weight = nn.Parameter(torch.ones(1,self.num_heads,1,1))
        self.neg_weight = nn.Parameter(torch.ones(1,self.num_heads,1,1))
        
    # def kernel(self, x):
    #     return torch.exp(torch.abs(x + self.inner_offset) - self.offset)

    def _attn(self, query, key, value, value2, attention_mask=None, head_mask=None):
        # scale = query.size(-1) ** 0.5
        # if self.normalized_attention:
        #     query = normalize(query)
        #     key = normalize(key)
        #     if self.softmax_temp is not None:
        #         query = query * self.softmax_temp

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value_min = torch.finfo(attn_weights.dtype).min
            # mask_value_max = torch.finfo(attn_weights.dtype).max
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value_min = torch.full([], mask_value_min, dtype=attn_weights.dtype, device=attn_weights.device)
            # mask_value_max = torch.full([], mask_value_max, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights_pos = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value_min)
            # attn_weights_neg = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value_max)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights_pos = attn_weights_pos + attention_mask

        #####
        # value_mask = (attn_weights > 0).to(attn_weights.dtype)
        attn_weights_pos = torch.exp(attn_weights_pos)
        attn_weights_neg = torch.where(causal_mask, 1 / (attn_weights_pos + 1e-6), torch.zeros_like(attn_weights_pos))
        attn_weights_pos = attn_weights_pos / torch.sum(attn_weights_pos, dim=-1, keepdim=True)
        attn_weights_neg = attn_weights_neg / torch.sum(attn_weights_neg, dim=-1, keepdim=True)


        # attn_weights_neg = torch.exp(attn_weights_neg * -1)
        # attn_weights_neg = attn_weights_neg / torch.sum(attn_weights_neg, dim=-1, keepdim=True)
        #####

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights_pos = attn_weights_pos.type(value.dtype)
        attn_weights_pos = self.attn_dropout(attn_weights_pos)
        attn_weights_neg = attn_weights_neg.type(value.dtype)
        attn_weights_neg = self.attn_dropout(attn_weights_neg)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights_pos = attn_weights_pos * head_mask

        attn_output_one = torch.matmul(attn_weights_pos, value)
        attn_output_two = torch.matmul(attn_weights_neg, value2)

        return attn_output_one, attn_output_two, attn_weights_pos

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        if not self.neg_version:
            query, key, value, value2 = self.c_attn(hidden_states).split(self.split_size, dim=2)
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            value2 = value * -1

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        value2 = self._split_heads(value2, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
            value2 = torch.cat((past_value2, value2), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output_one, attn_output_two, attn_weights = self._upcast_and_reordered_attn(query, key, value, value2, attention_mask, head_mask)
        else:
            attn_output_one, attn_output_two, attn_weights = self._attn(query, key, value, value2, attention_mask, head_mask)

        # attn_output = attn_output_one * value_mask + attn_output_two * (1 - value_mask)

        if self.c_proj_2 is not None:
            attn_output_one = self._merge_heads(attn_output_one, self.num_heads, self.head_dim)
            attn_output_one = self.c_proj(attn_output_one)
            attn_output_two = self._merge_heads(attn_output_two, self.num_heads, self.head_dim)
            attn_output_two = self.c_proj_2(attn_output_two)
            attn_output = attn_output_one + attn_output_two
        else:
            attn_output = attn_output_one * self.pos_weight + attn_output_two * self.neg_weight
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
            attn_output = self.c_proj(attn_output)
            
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


def patch_attn(model, offset=5, include_o=False, neg_version=False, inner_offset=0, alternate=False):
    conf = model.config
    idx = 0
    if alternate:
        indices = [1,3,5,7,9]
    else:
        indices = [n for n in range(30)]

    for n,m in model.named_modules():
        if hasattr(m, "attn"):
            if idx in indices:
                del m.attn
                m.add_module("attn", NewGPT2Attention(conf, is_cross_attention=False, layer_idx=None, offset=offset, include_o=include_o, neg_version=neg_version, inner_offset=inner_offset))
                print("activated", idx)
            idx += 1
            print('current idx', idx)


def patch_mlp(model, activation_type):
    idx = 0
    for n,m in model.named_modules():
        if hasattr(m, "mlp"):
            del m.mlp
            m.add_module("mlp", GPT2MLP(model.config, activation_type=activation_type[idx]))
            idx += 1


new_args = {
            "use_new_attn": True,
        "offset": 4,
        "neg_version": False,
        "include_o": False,
        "inner_offset": 0.0,
        "alternate": False,
        "activations": ["gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu"],
        # "activations": ["gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu"],
}
