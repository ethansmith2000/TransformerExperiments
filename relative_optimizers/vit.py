# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class BaselineAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., **kwargs):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_o = nn.Linear(inner_dim, dim, bias = True)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        # xavier init
        nn.init.xavier_normal_(self.to_q.weight)
        nn.init.xavier_normal_(self.to_k.weight)
        nn.init.xavier_normal_(self.to_v.weight)
        nn.init.xavier_normal_(self.to_o[0].weight)


    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q,k,v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class CorrelationAttention(BaselineAttention):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., **kwargs):
        super().__init__(dim, heads, dim_head, dropout)
        if kwargs.get("mod_q", False):
            self.to_q = nn.Identity()
        if kwargs.get("mod_k", False):
            self.to_k = nn.Identity()
        if kwargs.get("mod_v", False):
            self.to_v = nn.Identity()
        if kwargs.get("mod_o", False):
            self.to_o = nn.Sequential(
                nn.Identity(),
                nn.Dropout(dropout)
            )

def init_to_identity(module):
    if isinstance(module, nn.Linear):
        module.weight.data.fill_(0.0)
        module.weight.data[:, :] = torch.eye(module.weight.data.shape[1])

class CorrelationInitAttention(BaselineAttention):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., **kwargs):
        super().__init__(dim, heads, dim_head, dropout)
        if kwargs.get("mod_q", False):
            init_to_identity(self.to_q)
        if kwargs.get("mod_k", False):
            init_to_identity(self.to_k)
        if kwargs.get("mod_v", False):
            init_to_identity(self.to_v)
        if kwargs.get("mod_o", False):
            init_to_identity(self.to_o[0])

# class ResidualLinear(nn.Linear):
#     def forward(self, input):
#         return input + super().forward(input)

class ResidualLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, scale=1.0):
        super().__init__(in_features, out_features, bias)
        scale = torch.Tensor([scale])
        self.register_buffer("scale", scale)

    def forward(self, input):
        return input * self.scale + super().forward(input) * self.scale

class ResidualAttention(BaselineAttention):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., **kwargs):
        super().__init__(dim, heads, dim_head, dropout)
        if kwargs.get("mod_q", False):
            self.to_q = ResidualLinear(dim, dim, bias=False)
        if kwargs.get("mod_k", False):
            self.to_k = ResidualLinear(dim, dim, bias=False)
        if kwargs.get("mod_v", False):
            self.to_v = ResidualLinear(dim, dim, bias=False)
        if kwargs.get("mod_o", False):
            self.to_o = nn.Sequential(
                ResidualLinear(dim, dim, bias=True),
                nn.Dropout(dropout)
            )

def patch_model(model, exp_args):
    for name, m in model.named_modules():
        if hasattr(m, "attn"):
            if exp_args["mode"] == "correlation":
                m.attn = CorrelationAttention(model.config, **exp_args)
            elif exp_args["mode"] == "correlation_init":
                m.attn = CorrelationInitAttention(model.config, **exp_args)
            elif exp_args["mode"] == "residual":
                m.attn = ResidualAttention(model.config, **exp_args)
            elif exp_args["mode"] == "base":
                m.attn = BaselineAttention(model.config, **exp_args)
            elif exp_args["mode"] == "dummy":
                m.attn = nn.Identity()
            else:
                raise ValueError(f"Invalid mode: {exp_args['mode']}")
    return model, optimizer