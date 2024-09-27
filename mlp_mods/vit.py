# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from common.activations import Activation, LinearAct

class FeedForward(nn.Module):
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


class GegluFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., act="gelu", act_power=1):
        super().__init__()
        self.up = nn.Linear(dim, hidden_dim * 2)
        self.down = nn.Linear(hidden_dim, dim)
        self.act = Activation(act, power=act_power, dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x, gate = self.up(x).chunk(2, dim=-1)
        gate = self.act(gate)
        x = self.dropout(x)
        gate = self.dropout(gate)
        return self.dropout(self.down(x * gate))

class DoubleFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., act="gelu", act_power=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Activation(act, power=act_power, dim=hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            Activation(act, power=act_power, dim=hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def patch_model(model, args):
    for name, module in model.named_modules():
        if hasattr(module, 'ff'):
            if args.mode == "geglu":
                del module.ff
                module.add_module('ff', GegluFeedForward(module.dim, module.hidden_dim, args.dropout, args.activation, args.act_power))
            elif args.mode == "double":
                del module.ff
                module.add_module('ff', DoubleFeedForward(module.dim, module.hidden_dim, args.dropout, args.activation, args.act_power))

    # init weights again
    model.init_weights()
