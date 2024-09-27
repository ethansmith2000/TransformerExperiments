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


from torch.autograd import Function

class ReluForwardSiluBackward(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        sigmoid = torch.sigmoid(input)
        grad_input = grad_output * (sigmoid * (1 + input * (1 - sigmoid)))
        return grad_input

relu_fwd_silu_bwd = ReluForwardSiluBackward.apply

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
        elif activation_type == 'relu_fwd_silu_bwd':
            activation = relu_fwd_silu_bwd

        if power != 1.0:
            self.activation = lambda x: torch.pow(activation(x), power)
        else:
            self.activation = activation

    def forward(self, x):
        return self.activation(x)