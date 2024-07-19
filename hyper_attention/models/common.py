import torch
from torch import nn
import torch.nn.functional as F


class Activation(nn.Module):

    def __init__(self, activation_type: str = 'relu'):
        super().__init__()
        self.activation_type = activation_type
        if activation_type == 'relu':
            activation = lambda x: torch.nn.functional.relu(x)
        elif activation_type == 'gelu':
            activation = lambda x: torch.nn.functional.gelu(x)
        elif activation_type == 'silu':
            activation = lambda x: torch.nn.functional.silu(x)

        self.activation = activation

    def forward(self, x):
        return self.activation(x)


class HyperAttention(nn.Module):

    """
    HyperAttention. Do perceiver attention to create weight matrix that is (dim*2 x dim), chunk it
    then use it as an MLP with activation to inputs
    """

    def __init__(self, dim, heads=8, num_layers=2, qkv_bias=False, act_fn='gelu', attn_drop=0., proj_drop=0., out_bias=True):
        super().__init__()
        self.h = heads
        self.head_dim = dim // heads
        queries = [torch.nn.Parameter(torch.randn(1, dim, dim)) for _ in range(num_layers)]
        for q in queries:
            torch.nn.init.kaiming_uniform_(q)
        self.queries = torch.nn.Parameter(torch.cat(queries, dim=1))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=out_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act_fn = Activation(act_fn)
        self.num_layers = num_layers
        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        L = self.queries.shape[1]
        norm_attn_x = self.attn_norm(x)
        k, v = map(lambda t: t.reshape(B, N, self.h, self.head_dim).transpose(1, 2), self.kv(norm_attn_x).chunk(2, dim=-1))
        orig_layers = self.queries.expand(B, -1, -1)
        q = self.query_norm(orig_layers)
        q = q.reshape(B, L, self.h, self.head_dim).transpose(1, 2)

        layers = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, C)
        layers = self.proj_drop(self.proj(layers))

        layers = (orig_layers + layers).chunk(self.num_layers, dim=1)

        x = self.mlp_norm(x)
        for i in range(len(layers)):
            x = self.act_fn(torch.einsum('bnd,bld->bnl', x, layers[i]))

        return x



class HyperAttention2(nn.Module):

    def __init__(self, dim, act_fn='relu'):
        super().__init__()
        self.act_fn = Activation(act_fn)
        self.weight = torch.nn.Parameter(torch.randn(1, dim, dim) * 0.02)
        self.mean_proj = nn.Linear(dim, dim)
        self.std_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        mean, std = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)
        mean = self.mean_proj(mean)
        std = self.std_proj(std)

        weight1 = self.weight.expand(B, -1, -1) + mean
        weight2 = self.weight.expand(B, -1, -1) + std

        x = F.linear(x, weight1)
        x = self.act_fn(x)
        x = F.linear(x, weight2)

        return x


class HyperAttention3(nn.Module):

    def __init__(self, dim, act_fn='relu'):
        super().__init__()
        self.act_fn = Activation(act_fn)
        self.weight = torch.nn.Parameter(torch.randn(1, dim, dim) * 0.02)
        self.mean_proj = nn.Linear(dim, dim)
        self.std_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        mean, std = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)
        mean = self.mean_proj(mean)
        std = self.std_proj(std)

        weight1 = self.weight.expand(B, -1, -1) + mean
        weight2 = self.weight.expand(B, -1, -1) + std

        x = F.linear(x, weight1)
        x = self.act_fn(x)
        x = F.linear(x, weight2)

        return x