# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from common.activations import Activation, LinearAct

class AltActFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., act="gelu", act_power=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Activation(act, power=act_power, dim=hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class AltActAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., val_act=None, post_attn_act=None, power=1.0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        # self.val_act = nn.Identity() if val_act is None else Activation(val_act)
        # self.post_attn_act = nn.Identity() if post_attn_act is None else Activation(post_attn_act)
        self.val_act = LinearAct(inner_dim, inner_dim, activation_type=val_act, power=power, pre_act=True) if val_act is not None else nn.Identity()
        self.post_attn_act = LinearAct(dim, dim, activation_type=post_attn_act, power=power, pre_act=True) if post_attn_act is not None else nn.Identity()


    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        v = self.val_act(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q,k,v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.post_attn_act(out)
        return self.to_out(out)


def patch_model(model, args):
    for name, module in model.named_modules():
        if hasattr(module, 'attn'):
            module.attn = AltActAttention(args.dim, args.heads, args.dim_head, args.dropout, args.val_act, args.post_attn_act, args.act_power)
        if hasattr(module, 'ff'):
            module.ff = AltActFeedForward(args.dim, args.mlp_dim, args.dropout, args.act, args.act_power)

    # init weights again
    model.init_weights()
