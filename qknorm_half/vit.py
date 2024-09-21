import torch
from torch import nn
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention as sdpa

########

def normalize(tensor):
    eps = 1e-6 if tensor.dtype == torch.float16 else 1e-10
    norm = tensor.norm(dim=-1, keepdim=True)
    norm_clamped = torch.where(norm > eps, norm, eps)
    out = tensor / norm_clamped
    return out


class AttnNormBase(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.softmax_temp = torch.nn.Parameter(torch.ones(1, heads, 1, 1) * 10)

    def forward(self, x):
        raise NotImplementedError

class KNormAttention(AttnNormBase):

    def forward(self, x):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (self.to_qkv(x).chunk(3, dim = -1)))
        k = normalize(k) * self.softmax_temp
        out = sdpa(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class QNormAttention(AttnNormBase):

    def forward(self, x):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (self.to_qkv(x).chunk(3, dim = -1)))
        q = normalize(q) * self.softmax_temp
        out = sdpa(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class QKNormAttention(AttnNormBase):

    def forward(self, x):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (self.to_qkv(x).chunk(3, dim = -1)))
        q = normalize(q)
        k = normalize(k) * self.softmax_temp
        out = sdpa(q, k, v, scale=1.0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def patch(vit, mode="knorm"):
    for i in range(len(vit.transformer.attns)):
        lyr = vit.transformer.attns[i]
        if mode == "knorm":
            vit.transformer.attns[i] = KNormAttention(dim=lyr.to_qkv.in_features, heads=lyr.heads, dropout=lyr.to_out[1].p)
        elif mode == "qnorm":
            vit.transformer.attns[i] = QNormAttention(dim=lyr.to_qkv.in_features, heads=lyr.heads, dropout=lyr.to_out[1].p)
        elif mode == "qknorm":
            vit.transformer.attns[i] = QKNormAttention(dim=lyr.to_qkv.in_features, heads=lyr.heads, dropout=lyr.to_out[1].p)
        else:
            raise ValueError(f"Unknown mode: {mode}")


extra_args = {
    "mode": "knorm"
}

def get_run_name():
    if args.mode == "knorm":
        watermark = "k_" + watermark
    elif args.mode == "qknorm":
        watermark = "qk_" + watermark
    elif args.mode == "qnorm":
        watermark = "q_" + watermark