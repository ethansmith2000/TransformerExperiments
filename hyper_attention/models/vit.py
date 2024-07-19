# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .common import HyperAttention

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
    def __init__(self, dim, hidden_dim, dropout = 0., act="gelu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, 
                dim=512, 
                heads = 8, 
                dropout = 0., 
                post_attn_act=False,
                ):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Dropout(dropout)
        )
        self.post_attn_act = nn.Identity()
        if post_attn_act:
            self.post_attn_act = nn.Sequential(nn.Linear(dim, dim), nn.GELU())


    def forward(self, x):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (self.to_qkv(x).chunk(3, dim = -1)))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.post_attn_act(out)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, 
                        depth, 
                        heads, 
                        mlp_dim, 
                        dropout = 0., 
                        act="gelu", 
                        post_attn_act=None, 
                        hyper_attn=False
                        ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, post_attn_act=post_attn_act)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout, act=act)) if not hyper_attn else HyperAttention(dim, heads=heads, proj_drop=dropout, attn_drop=dropout)
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
                    channels = 3, 
                    dropout = 0., 
                    emb_dropout = 0.,
                    
                    act="gelu",
                    post_attn_act=None,
                    hyper_attn=False

                    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, 
                                    dropout, act=act, post_attn_act=post_attn_act,
                                    hyper_attn=hyper_attn)


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

        x = x[:, 0]

        return x

    def forward(self, img):
        x = self.get_feat(img)
        return self.mlp_head(x)
