# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

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
            self.activation = torch.sin

    def forward(self, x):
        return self.activation(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., activation_type = 'gelu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Activation(activation_type),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., dipole_attention=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.dipole_attention = dipole_attention

        if self.dipole_attention:
            self.to_v_2 = nn.Linear(dim, inner_dim, bias = False)
            self.pos_weight = nn.Parameter(torch.randn(1, heads, 1, 1))
        else:
            self.to_v_2 = None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if self.to_v_2 is not None:
            v2 = rearrange(self.to_v_2(x), 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        if self.dipole_attention:
            neg_attn = 1 / (attn + 1e-9)
            neg_attn = neg_attn / (neg_attn.sum(dim=-1, keepdim=True))
            out1 = torch.matmul(attn, v)
            out2 = torch.matmul(neg_attn, v2)
            # pos_weight = torch.sigmoid(self.pos_weight)
            pos_weight = self.pos_weight
            neg_weight = 1 - pos_weight
            out = pos_weight * out1 + neg_weight * out2
        else:
            out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., activations=[], dipole_attention=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        if len(activations) < depth:
            activations = ['gelu'] * (depth)

        da_inds = [False] * depth
        if dipole_attention is not None:
            if dipole_attention == "even_alt":
                da_inds = [True, False] * (depth // 2)
            elif dipole_attention == "odd_alt":
                da_inds = [False, True] * (depth // 2)
            elif dipole_attention == "all":
                da_inds = [True] * depth
            else:
                raise ValueError("Invalid dipole_attention value")

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, dipole_attention=da_inds[i])),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, activation_type=activations[i]))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., activations=[], dipole_attention=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, activations, dipole_attention=dipole_attention)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
