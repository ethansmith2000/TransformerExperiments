import torch
from torch import nn
import torch.nn.functional as F

#https://github.com/ethansmith2000/SparseNetworks

class Relu2(nn.Module):
    def forward(self, x):
        return F.relu(x).square()

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == "relu2":
        return Relu2()
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")

class PermuteIn(nn.Module):

    def __init__(self, 
                full_dim,
                heads,
                mode="structured", # random, roll, chunk_random, structured
                roll=0.4,
                chunks=4, # must divide the chunk dim evenly
                ):
        super().__init__()
        block_dim = full_dim // heads
        roll = int(roll * full_dim)
        if mode == "random":
            permute = torch.randperm(full_dim)
        elif mode == "roll":
            permute = torch.roll(torch.arange(full_dim), roll)
        elif mode == "chunk_random":
            assert block_dim % chunks == 0, "chunks must divide the dim evenly"
            chunk_indices = torch.randperm(full_dim // (block_dim // chunks))
            permute = torch.cat([torch.arange((block_dim // chunks)) + i * (block_dim // chunks) for i in chunk_indices])
        elif mode == "structured":
            indices = torch.arange(full_dim)
            permute = (indices % heads) * block_dim + (indices // heads)
        else:
            raise NotImplementedError("mode not implemented")
        self.register_buffer("permute", permute)

    def forward(self, x):
        return x[:, self.permute]


class Unpermute(nn.Module):

    def __init__(self, indices):
        super().__init__()
        perm_matrix = F.one_hot(indices, num_classes=indices.shape[0]).float()
        unperm_matrix = perm_matrix.inverse()
        unperm = unperm_matrix.argmax(dim=-1).long()
        self.register_buffer("unperm", unperm)

    def forward(self, x):
        return x[:, self.unperm]



class AddBias(nn.Module):

    def __init__(self, dim):
        super(AddBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.bias


class LowRankLinear(nn.Module):
    """
    Like LoRA but without base layer
    """
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LowRankLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.weight1 = nn.Parameter(torch.Tensor(in_features, rank))
        self.weight2 = nn.Parameter(torch.Tensor(rank, out_features))
        self.add_bias = AddBias(out_features) if bias else torch.nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=5**0.5)
        nn.init.kaiming_uniform_(self.weight2, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.add_bias(torch.mm(torch.mm(x, self.weight1), self.weight2))


class SparseLinear(nn.Module):

    """
    Kinda like Monarch Matrices I think
    """
    
    def __init__(self, full_in_dim=1024, full_out_dim=1024, heads=8, bias=True):
        super(SparseLinear, self).__init__()
        self.full_in = full_in_dim
        self.full_out = full_out_dim
        self.in_dim = full_in_dim // heads
        self.out_dim = full_out_dim // heads
        self.h = heads
        weights = [torch.randn(self.in_dim, self.out_dim) for _ in range(heads)]
        for i in range(len(weights)):
            torch.nn.init.kaiming_uniform_(weights[i], gain=torch.nn.init.calculate_gain('relu'))
        self.weight = nn.Parameter(torch.stack(weights, dim=0))
        self.bias_add = AddBias(self.full_out) if bias else nn.Identity()

    def forward(self, x):
        b, h, in_dim = x.shape[0], self.h, self.in_dim
        x = x.reshape(b, h, in_dim)
        x = torch.einsum('bhd,hdl->bhl', x, self.weight)
        x = x.reshape(b, h * self.out_dim)
        x = self.bias_add(x)
        return x



class SparseMLPResidual(nn.Module):
    """
    permute/unpermute operation to align with residual stream
    """

    def __init__(self, full_dim=1024, 
                        heads=8, 
                        act="gelu", 
                        full_mlp_dim=4096, 
                        unperm=True, 
                        dropout=0., 
                        permute_mode="structured", # ["random", "roll", "chunk_random", "linear", "structured"]
                        bias=True
                        ):
        super().__init__()
        self.up = SparseLinear(full_dim, full_mlp_dim, heads, bias=bias)
        self.down = SparseLinear(full_mlp_dim, full_dim, heads, bias=bias)
        self.act = get_activation(act) if act is not None else nn.Identity()

        self.unperm = nn.Identity()
        if permute_mode != "linear":
            self.perm = PermuteIn(full_dim, heads, mode=permute_mode)
            if unperm:
                self.unperm = Unpermute(self.perm.permute)
        else:
            self.perm = nn.Linear(full_dim, full_dim)
            if unperm:
                self.unperm = nn.Linear(full_dim, full_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.perm(x) # reorder features to have different interactions
        x = self.up(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.down(x)
        x = self.dropout(x)
        x = self.unperm(x)

        return x

class SparseFeedForward(SparseMLPResidual):

    def forward(self, x):
        b, toks, d = x.shape
        x = x.reshape(b * toks, d)
        x = super().forward(x)
        x = x.reshape(b, toks, d)
        return x



class SparseMLP(nn.Module):
    """
    Closer to how monarch matrices does it i think
    """

    def __init__(self, full_dim=1024, 
                        heads=8, 
                        act="gelu",
                        full_mlp_dim=4096, 
                        dropout=0., 
                        permute_mode="structured", # ["random", "roll", "chunk_random", "linear", "structured"]
                        bias=True
                        ):
        super().__init__()
        self.up = SparseLinear(full_dim, full_mlp_dim, heads, bias=bias)
        self.down = SparseLinear(full_mlp_dim, full_dim, heads, bias=bias)
        self.act = get_activation(act) if act is not None else nn.Identity()

        if permute_mode != "linear":
            self.perm = PermuteIn(full_mlp_dim, heads, mode=permute_mode)
        else:
            self.perm = nn.Linear(full_mlp_dim, full_mlp_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.perm(x)
        x = self.down(x)
        x = self.dropout(x)

        return x


