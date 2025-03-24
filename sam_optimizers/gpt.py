import torch
from torch import nn
from typing import Optional, Tuple
from einops import rearrange
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa
from .adam_two_momentum_perturb import AdamTwoMomentumSAM
from .adam_wd_perturb import AdamWeightDecaySAM
from .muon_adam_perturb import MuonAdamSAM
from .nesterov_perturb import NesterovPerturb
from hf_gpt_blocks.hf_gpt import patch_gpt
from .muon import Muon


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
    if exp_args["mode"] == "adam_two_momentum_perturb":
        optimizer = AdamTwoMomentumSAM(optimizer_grouped_parameters, beta1=args.beta1, beta2=args.beta2, lr=args.learning_rate, perturb_lr_ratio=exp_args["perturb_lr_ratio"], beta1_perturb=exp_args["beta1_perturb"], eps=args.eps)
    elif exp_args["mode"] == "adam_wd_perturb":
        optimizer = AdamWeightDecaySAM(optimizer_grouped_parameters, beta1=args.beta1, beta2=args.beta2, lr=args.learning_rate, eps=args.eps)
    elif exp_args["mode"] == "muon_adam_perturb":
        optimizer = MuonAdamSAM(optimizer_grouped_parameters, beta1=args.beta1, beta2=args.beta2, lr=args.learning_rate, perturb_lr_ratio=exp_args["perturb_lr_ratio"], eps=args.eps)
    elif exp_args["mode"] == "nesterov_perturb":
        optimizer = NesterovPerturb(optimizer_grouped_parameters, beta1=args.beta1, beta2=args.beta2, lr=args.learning_rate, perturb_lr_ratio=exp_args["perturb_lr_ratio"], eps=args.eps)
    elif exp_args["mode"] == "muon":
        optimizer = Muon(optimizer_grouped_parameters, beta1=args.beta1, beta2=args.beta2, lr=args.learning_rate, eps=args.eps)
    elif exp_args["mode"] == "adam":
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps, fused=not args.compile_optimizer)
    else:
        raise ValueError(f"Invalid optimizer: {exp_args['mode']}")

    return optimizer


def patch_model(model, args, exp_args):
    patch_gpt(model)
    return model


def get_run_name(args, exp_args):
    run_name = exp_args["mode"]
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"

    return args, run_name


extra_args = {
    # "mode": "adam", # ["adam", "muon", "nesterov_perturb", "adam_two_momentum_perturb", "adam_wd_perturb", "muon_adam_perturb"]
    # "mode": "adam",

    # "mode": "muon",

    "mode": "adam_two_momentum_perturb",
    "perturb_lr_ratio": 1.0,
    "beta1_perturb": 0.95,

    # "mode": "adam_wd_perturb",

    # "mode": "muon_adam_perturb",
    # "perturb_lr_ratio": 1e-4,

    # "mode": "nesterov_perturb",
    # "perturb_lr_ratio": 1e-4,

    
    
    
}
