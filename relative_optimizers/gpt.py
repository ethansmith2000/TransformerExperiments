import torch
from .relative_adam import RelativeAdam
from .relative_adam_2 import RelativeAdam2
from torch import nn
from typing import Optional, Tuple
from einops import rearrange
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa

class AttentionBase(nn.Module):
    """
    Causal multihead attention that uses torch's SDPA 
    """
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim*3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,):

        b, n, d, h = (*x.shape, self.heads)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (self.to_qkv(x).chunk(3, dim=-1)))
        outputs = (self.to_out(rearrange(sdpa(q, k, v, is_causal=True), 'b h n d -> b n (h d)')), None)
        return outputs


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
    if exp_args["mode"] == "relative_adam":
        optimizer = RelativeAdam(optimizer_grouped_parameters, beta1=args.beta1, beta2=args.beta2, lr_weight=exp_args["lr_weight"], param_lr=exp_args["param_lr"], param_eps=exp_args["param_eps"], eps=args.eps)
    elif exp_args["mode"] == "relative_adam_2":
        optimizer = RelativeAdam2(optimizer_grouped_parameters, param_lr=exp_args["param_lr"], beta1=args.beta1, beta2=args.beta2, param_eps=exp_args["param_eps"], eps=args.eps)
    elif exp_args["mode"] == "base":
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps, fused=not args.compile_optimizer)
    else:
        raise ValueError(f"Invalid optimizer: {exp_args['mode']}")

    return optimizer


# def patch_model(model, args, exp_args):
#     # just to make things run faster
#     for n,m in model.named_modules():
#         if hasattr(m, "attn"):
#             dim = model.config.n_embd
#             heads = m.attn.num_heads

#             m.attn = AttentionBase(dim=dim, heads=heads)


def get_run_name(args, exp_args):
    if exp_args["mode"] == "base":
        run_name = "base"
    elif exp_args["mode"] == "relative_adam":
        run_name = "mode_" + exp_args["mode"]  + "_p_eps_" + str(exp_args["param_eps"]) + "_p_lr_" + str(exp_args["param_lr"]) + "_lr_w_" + str(exp_args["lr_weight"]) + "_lr_" + str(args["learning_rate"])
    elif exp_args["mode"] == "relative_adam_2":
        run_name = "mode_" + exp_args["mode"]  + "_p_eps_" + str(exp_args["param_eps"]) + "_lr_" + str(args["learning_rate"])
    else:
        raise ValueError(f"Invalid optimizer: {exp_args['mode']}")
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"

    return args, run_name


extra_args = {
    # "mode": "base", # ["base", "relative_adam", "relative_adam_2"]

    "mode": "relative_adam",
    "param_eps": 1e-4,
    "lr_weight": 0.5,
    "param_lr": 0.005,

    # "mode": "relative_adam_2",
    # "param_eps": 1e-4,
}
