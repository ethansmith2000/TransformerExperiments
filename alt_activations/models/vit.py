# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .common import Activation, LinearAct

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

class Attention(nn.Module):
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

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., acts="gelu", act_powers=1, val_act=None, post_attn_act=None, attn_power=1.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        acts = [acts] * depth if isinstance(acts, str) else acts
        act_powers = [act_powers] * depth if (isinstance(act_powers, int) or isinstance(act_powers, float)) else act_powers
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, val_act=val_act, post_attn_act=post_attn_act, power=attn_power)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout, acts[i], act_powers[i]))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, 
                    image_size, 
                    patch_size, 
                    num_classes, 
                    dim, 
                    depth, 
                    heads, 
                    mlp_dim, 
                    pool = 'cls', 
                    channels = 3, 
                    dim_head = 64, 
                    dropout = 0., 
                    emb_dropout = 0.,
                    acts="gelu",
                    act_powers=1,
                    val_act=None,
                    post_attn_act=None,
                    attn_power=1.0,
                    ):
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

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, acts=acts, act_powers=act_powers, val_act=val_act, post_attn_act=post_attn_act, attn_power=attn_power)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def get_feat(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x

    def forward(self, img):
        x = self.get_feat(img)
        return self.mlp_head(x)
