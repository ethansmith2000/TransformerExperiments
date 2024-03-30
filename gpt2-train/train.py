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

def normalize(tensor):
    # eps = torch.finfo(tensor.dtype).eps
    eps = 1e-6
    norm = tensor.norm(dim=-1, keepdim=True)
    norm_clamped = torch.where(norm > eps, norm, eps)
    out = tensor / norm_clamped
    return out


# scale = None # auto 1/sqrt(d)
# if self.normalized_attention:
#     # print("CROSS QUERY BEFORE NORM NAN", torch.isnan(query).any(), torch.isinf(query).any())
#     query = normalize(query)
#     key = normalize(key)
#     scale = 10 # fixed scale based on what others have done
#     if self.softmax_temp is not None:
#         # print("CROSS QUERY AFTER NORM NAN", torch.isnan(query).any(), torch.isinf(query).any())
#         query = query * self.softmax_temp
#         scale = 1.0 # no multiplier because we're doing our own by multiplying keys/values

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





# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.39.0.dev0")

logger = get_logger(__name__)

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



def main():

    args = {
        "use_new_attn": True,
        "offset": 4,
        "neg_version": False,
        "include_o": False,
        "inner_offset": 0.0,
        "alternate": False,
        "activations": ["gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu"],
        # "activations": ["gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu"],


        "num_validation_batches": 25,
        "validate_every": 1000,
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-103-v1",
        "train_file": None,
        "validation_file": None,
        "validation_split_percentage": 5,
        # "model_name_or_path": "openai-community/gpt2-medium",
        "model_name_or_path": "openai-community/gpt2",
        "config_name": None,
        "tokenizer_name": None,
        "use_slow_tokenizer": False,
        "per_device_train_batch_size": 28,
        "learning_rate": 5.0e-5,
        "weight_decay": 0.01,
        "num_train_epochs": 2,
        "max_train_steps": None,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 250,
        "seed": 123,
        "model_type": None,
        "block_size": None,
        "preprocessing_num_workers": 10,
        "overwrite_cache": False,
        "no_keep_linebreaks": False,
        "trust_remote_code": False,
        "checkpointing_steps": None,
        "resume_from_checkpoint": None,
        "with_tracking": True,
        "report_to": "wandb",
        "low_cpu_mem_usage": False,
        # "max_grad_norm": None,
        "max_grad_norm": 0.2,
        "hf_path": None,
        "base_output_dir": None,
    }


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

    args = SimpleNamespace(**args)

    print("Running with the following arguments:")
    print(json.dumps(vars(args), indent=2))

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.output_dir is None:
        args.output_dir = time.strftime("run_%Y%m%d_%H%M%S")

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, 
                                                            mixed_precision="fp16",
                                                            **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.hf_path)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.hf_path
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.hf_path
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=args.hf_path, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.hf_path
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.hf_path
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        # model = AutoModelForCausalLM.from_pretrained(
        #     args.model_name_or_path,
        #     # from_tf=bool(".ckpt" in args.model_name_or_path),
        #     config=config,
        #     low_cpu_mem_usage=args.low_cpu_mem_usage,
        #     trust_remote_code=args.trust_remote_code,
        # )

        model = AutoModelForCausalLM.from_config(
            config,
            # args.model_name_or_path,
            # from_tf=bool(".ckpt" in args.model_name_or_path),
            # config=config,
            # low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    model.gradient_checkpointing_enable()

    if args.use_new_attn:
        patch_attn(model, offset=args.offset, include_o=args.include_o, neg_version=args.neg_version, inner_offset=args.inner_offset, alternate=args.alternate)
    
    if len(args.activations) > 0:
        patch_mlp(model, args.activations)

    print(model)


    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        init_kwargs = {
            "wandb":
                {
                    "name": f"{base_str}",
                }
        }
        accelerator.init_trackers("clm_no_trainer", experiment_config, init_kwargs=init_kwargs)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            model.train()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip the gradients
                mini_logs ={
                        "step_loss": loss.detach().float(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                if args.max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    mini_logs["grad_norm"] = grad_norm
                
                accelerator.log(
                        mini_logs,
                        step=completed_steps,
                    )

                if step % 10 == 0:
                    profile_gpus()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

            
            if completed_steps % args.validate_every == 0:
                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)

                    loss = outputs.loss
                    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_train_batch_size)))
                    if args.num_validation_batches is not None:
                        if step >= args.num_validation_batches:
                            break

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

                if args.with_tracking:
                    accelerator.log(
                        {
                            "perplexity": perplexity,
                            "eval_loss": eval_loss,
                            "train_loss": total_loss.item() / len(train_dataloader),
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        print("Saving model to", args.output_dir)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
