import torch
from torch import nn
from typing import Optional, Tuple
from .attentions import (
    AttentionBase,
    AttentionRelativeScaling1,
    AttentionRelativeScaling2,
    AttentionYarnScaling,
    AttentionPolyFitScaling,
    AttentionLearnedScaling,
    AttentionSoftmaxPlusOne,
    AttentionSoftmaxPlusFN
)
def patch_model(model, args, exp_args):
    for n,m in model.named_modules():
        if hasattr(m, "attn"):
            dim = model.config.n_embd
            heads = m.attn.num_heads
            
            if exp_args["mode"] == "base":
                m.attn = AttentionBase(dim=dim, heads=heads)
            elif exp_args["mode"] == "relative_scaling_1":
                m.attn = AttentionRelativeScaling1(dim=dim, heads=heads, base_seq_len=exp_args["base_seq_len"])
            elif exp_args["mode"] == "relative_scaling_2":
                m.attn = AttentionRelativeScaling2(dim=dim, heads=heads, base_seq_len=exp_args["base_seq_len"], attn_bias=exp_args["attn_bias"], learned_bias=exp_args["learned_bias"])
            elif exp_args["mode"] == "yarn_scaling":
                m.attn = AttentionYarnScaling(dim=dim, heads=heads)
            elif exp_args["mode"] == "polyfit_scaling":
                m.attn = AttentionPolyFitScaling(dim=dim, heads=heads)
            elif exp_args["mode"] == "learned_scaling":
                m.attn = AttentionLearnedScaling(dim=dim, heads=heads, **exp_args)
            elif exp_args["mode"] == "softmax_plus_one":
                m.attn = AttentionSoftmaxPlusOne(dim=dim, heads=heads, **exp_args)
            elif exp_args["mode"] == "softmax_plus_fn":
                m.attn = AttentionSoftmaxPlusFN(dim=dim, heads=heads)
            else:
                raise ValueError(f"Invalid mode: {exp_args['mode']}")

    return model
            


def get_run_name(args, exp_args):
    if exp_args["mode"] in ["base", "yarn_scaling", "polyfit_scaling", "learned_scaling", "softmax_plus_one", "softmax_plus_fn"]:
        run_name = f"{exp_args['mode']}_lr:{args['learning_rate']}"
    elif exp_args["mode"] in ["relative_scaling_1", "relative_scaling_2"]:
        run_name = f"{exp_args['mode']}_base_seq_len:{exp_args['base_seq_len']}_lr:{args['learning_rate']}"
    elif exp_args["mode"] in ["relative_scaling_2"]:
        run_name = f"{exp_args['mode']}_base_seq_len:{exp_args['base_seq_len']}_attn_bias:{exp_args['attn_bias']}_learned_bias:{exp_args['learned_bias']}_lr:{args['learning_rate']}"
    else:
        raise ValueError(f"Invalid mode: {exp_args['mode']}")
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"

    return args, run_name


extra_args = {
    # base
    "mode": "base",

    # # relative scaling
    # "mode": "relative_scaling_1",
    # "base_seq_len": 2048,

    # # relative scaling 2
    # "mode": "relative_scaling_2",
    # "base_seq_len": 2048,
    # "attn_bias": 1.5,
    # "learned_bias": False,

    # # softmax plus one
    # "mode": "softmax_plus_one",

    # # softmax plus fn
    # "mode": "softmax_plus_fn",

    # # polyfit scaling
    # "mode": "polyfit_scaling",

    # # learned scaling
    # "mode": "learned_scaling",

    # # softmax plus one
    # "mode": "softmax_plus_one",

    # # softmax plus fn
    # "mode": "softmax_plus_fn",
    
    
}
