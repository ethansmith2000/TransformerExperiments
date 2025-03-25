import torch
from .relative_adam import RelativeAdam
from .muon import Muon
from .relative_muon import RelativeMuon
from .relative_muon_2 import RelativeMuon2
from torch import nn
from typing import Optional, Tuple
from einops import rearrange
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa
from hf_gpt_blocks.hf_gpt import patch_gpt


def patch_optimizer(model, args, exp_args):
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

    lr = exp_args.get("lr", args.learning_rate)
    weight_decay = exp_args.get("weight_decay", args.weight_decay)
    beta1 = exp_args.get("beta1", args.beta1)
    beta2 = exp_args.get("beta2", args.beta2)
    eps = exp_args.get("eps", args.eps)

    if exp_args["mode"] == "relative_adam":
        optimizer = RelativeAdam(optimizer_grouped_parameters, weight_decay=weight_decay, beta1=beta1, beta2=beta2, lr=lr, lr_weight=exp_args["lr_weight"], param_lr=exp_args["param_lr"], param_eps=exp_args["param_eps"], eps=eps)
    elif exp_args["mode"] == "adam":
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=(beta1, beta2), eps=eps, fused=not args.compile_optimizer, weight_decay=weight_decay)
    elif exp_args["mode"] == "muon":
        optimizer = Muon(optimizer_grouped_parameters, lr=lr, beta1=beta1, eps=eps, weight_decay=weight_decay)
    elif exp_args["mode"] == "relative_muon":
        optimizer = RelativeMuon(optimizer_grouped_parameters, lr=lr, beta1=beta1, eps=eps, weight_decay=weight_decay, param_lr=exp_args["param_lr"], param_eps=exp_args["param_eps"], lr_weight=exp_args["lr_weight"], lr_cap=exp_args["lr_cap"])
    elif exp_args["mode"] == "relative_muon_2":
        optimizer = RelativeMuon2(optimizer_grouped_parameters, lr=lr, beta1=beta1, eps=eps, weight_decay=weight_decay, param_lr=exp_args["param_lr"], lr_weight=exp_args["lr_weight"])
    else:
        raise ValueError(f"Invalid optimizer: {exp_args['mode']}")

    return optimizer


def patch_model(model, args, exp_args):
    patch_gpt(model)
    return model


def get_run_name(args, exp_args):
    run_name = exp_args["mode"]

    # if exp_args["mode"] == "base":
    #     run_name = "base"
    # elif exp_args["mode"] == "relative_adam":
    #     run_name = "mode_" + exp_args["mode"]  + "_p_eps_" + str(exp_args["param_eps"]) + "_p_lr_" + str(exp_args["param_lr"]) + "_lr_w_" + str(exp_args["lr_weight"]) + "_lr_" + str(args["learning_rate"])
    # elif exp_args["mode"] == "relative_adam_2":
    #     run_name = "mode_" + exp_args["mode"]  + "_p_eps_" + str(exp_args["param_eps"]) + "_lr_" + str(args["learning_rate"])
    # else:
    #     raise ValueError(f"Invalid optimizer: {exp_args['mode']}")
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"

    return args, run_name


extra_args = {
    # "mode": "adam", # ["base", "relative_adam", "relative_adam_2"]
    # "lr": 5.0e-5,
    # "weight_decay": 0.01,
    # "beta1": 0.9,
    # "beta2": 0.99,

    "mode": "relative_adam",
    "lr": 5.0e-5,
    "weight_decay": 0.2,
    "beta1": 0.9,
    "beta2": 0.98,
    "param_eps": 1e-4,
    "lr_weight": 0.4,
    "param_lr": 0.01,
    "lr_cap": 0.01,

    # "mode": "muon",
    # "lr": 2.0e-3,
    # "weight_decay": 0.01,
    # "beta1": 0.95,


    # "mode": "relative_muon",
    # "lr": 2.0e-3,
    # "weight_decay": 0.1,
    # "beta1": 0.95,
    # "param_lr": 0.1,
    # "param_eps": 1e-4,
    # "lr_weight": 0.5,
    # "lr_cap": 0.5,

    # "mode": "relative_muon_2",
    # "lr": 2.0e-3,
    # "weight_decay": 0.1,
    # "beta1": 0.95,
    # "param_lr": 0.1,
    # "lr_weight": 0.5,

}
