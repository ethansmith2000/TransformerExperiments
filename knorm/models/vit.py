# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



import torch
from torch import nn
import torch.nn.functional as F


class SignGelu2(nn.Module):

    def __init__(self, neg_scale=20.0):
        super().__init__()
        self.neg_scale = neg_scale

    def forward(self, x):
        sign = torch.sign(x)
        sign = torch.where(sign < 0, sign * self.neg_scale, sign)
        return sign * F.gelu(x).square()


class SinLU(nn.Module):
    def __init__(self, dim=None):
        super(SinLU,self).__init__()
        dim = 1 if dim is None else dim
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.ones(dim))
    def forward(self,x):
        return torch.sigmoid(x)*(x+self.a*torch.sin(self.b*x))


class NormalizedExp(nn.Module):

    def __init__(self, beta=0.99):
        super().__init__()
        self.register_buffer("avg_max", torch.tensor(1.0))
        self.beta = beta

    def forward(self, x):
        max_val = torch.max(x) / 2
        self.avg_max = self.beta * self.avg_max + (1 - self.beta) * max_val
        return torch.exp(x - self.avg_max)


class LinearAct(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, activation_type='relu', power=1.0, pre_act=False):
        super().__init__()
        self.pre_act = Activation(activation_type, power, dim=in_features) if pre_act else nn.Identity()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.post_act = Activation(activation_type, power, dim=out_features) if not pre_act else nn.Identity()

    def forward(self, x):
        return self.post_act(self.linear(self.pre_act(x)))


class Activation(nn.Module):

    def __init__(self, activation_type: str = 'relu', power=1.0, dim=None):
        super().__init__()
        self.activation_type = activation_type
        if activation_type == 'relu':
            activation = lambda x: torch.nn.functional.relu(x)
        elif activation_type == 'gelu':
            activation = lambda x: torch.nn.functional.gelu(x)
        elif activation_type == 'silu':
            activation = lambda x: torch.nn.functional.silu(x)
        elif activation_type == 'tanh':
            activation = lambda x: torch.tanh(x)
        elif activation_type == 'leaky':
            activation = lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        elif activation_type == 'sin':
            activation = lambda x: torch.sin(x)
        elif activation_type == 'sin_residual':
            activation = lambda x: torch.sin(x) + (x/2)
        elif activation_type == 'relu_sin':
            activation = lambda x: torch.sin(torch.relu(x)) + torch.relu(x/2)
        elif activation_type == 'norm_exp':
            activation = NormalizedExp()
        elif activation_type == 'sign_gelu2':
            activation = SignGelu2(neg_scale=20.0)
        elif activation_type == 'sinlu':
            activation = SinLU(dim=dim)

        if power != 1.0:
            self.activation = lambda x: torch.pow(activation(x), power)
        else:
            self.activation = activation

    def forward(self, x):
        return self.activation(x)
        

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., act="gelu", act_power=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Activation(act, power=act_power),
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


    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q,k,v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., acts="gelu", act_powers=1, val_act=None, post_attn_act=None, attn_power=1.0):
        super().__init__()
        self.attn_norms = nn.ModuleList([])
        self.ff_norms = nn.ModuleList([])
        self.attns = nn.ModuleList([])
        self.ffs = nn.ModuleList([])
        acts = [acts] * depth if isinstance(acts, str) else acts
        act_powers = [act_powers] * depth if (isinstance(act_powers, int) or isinstance(act_powers, float)) else act_powers
        for i in range(depth):
            self.attn_norms.append(nn.LayerNorm(dim))
            self.ff_norms.append(nn.LayerNorm(dim))
            self.attns.append(Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, val_act=val_act, post_attn_act=post_attn_act, power=attn_power))
            self.ffs.append(FeedForward(dim, mlp_dim, dropout, acts[i], act_powers[i]))
    def forward(self, x):
        for attn_norm, attn, ff_norm, ff in zip(self.attn_norms, self.attns, self.ff_norms, self.ffs):
            x = attn(attn_norm(x)) + x
            x = ff(ff_norm(x)) + x
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





########

def normalize(tensor):
    eps = 1e-6 if tensor.dtype == torch.float16 else 1e-10
    norm = tensor.norm(dim=-1, keepdim=True)
    norm_clamped = torch.where(norm > eps, norm, eps)
    out = tensor / norm_clamped
    return out


class KNormAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.softmax_temp = torch.nn.Parameter(torch.ones(1, heads, 1, 1) * 10)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q,k,v))
        k = normalize(k) * self.softmax_temp
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class QKNormAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.softmax_temp = torch.nn.Parameter(torch.ones(1, heads, 1, 1) * 10)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q,k,v))
        q = normalize(q)
        k = normalize(k) * self.softmax_temp
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def patch(vit, mode="knorm"):
    for i in range(len(vit.transformer.attns)):
        lyr = vit.transformer.attns[i]
        if mode == "knorm":
            vit.transformer.attns[i] = KNormAttention(dim=lyr.to_qkv.in_features, heads=lyr.heads)
        else:
            vit.transformer.attns[i] = QKNormAttention(dim=lyr.to_qkv.in_features, heads=lyr.heads)