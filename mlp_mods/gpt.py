import torch
from torch import nn
from typing import Optional, Tuple
from common.activations import Activation


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class GPT2MLP(nn.Module):
    def __init__(self, config, mlp_mult=4, activation_type='gelu', power=1.0):
        super().__init__()
        embed_dim = config.hidden_size
        intermediate_size = embed_dim * mlp_mult
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = Activation(activation_type=activation_type, power=power)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2MLPGeGLU(nn.Module):
    def __init__(self, config, mlp_mult=4, activation_type='gelu', power=1.0):
        super().__init__()
        embed_dim = config.hidden_size
        intermediate_size = embed_dim * mlp_mult
        self.c_fc = Conv1D(intermediate_size * 2, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = Activation(activation_type=activation_type, power=power)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states, gate = self.c_fc(hidden_states).chunk(2, dim=-1)
        hidden_states = self.act(gate) * hidden_states
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states



class GPT2MLPDouble(nn.Module):
    def __init__(self, config, mlp_mult=4, activation_type='gelu', power=1.0):
        super().__init__()
        embed_dim = config.hidden_size
        intermediate_size = embed_dim * mlp_mult
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_middle = Conv1D(intermediate_size, intermediate_size)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = Activation(activation_type=activation_type, power=power)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_middle(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states



def patch_model(model, exp_args):
    idx = 0
    for n,m in model.named_modules():
        if hasattr(m, "mlp"):

            meets_criteria = False
            if exp_args["targets"] == "all":
                meets_criteria = True
            elif exp_args["targets"] == "even":
                meets_criteria = idx % 2 == 0
            elif exp_args["targets"] == "odd":
                meets_criteria = idx % 2 == 1
            elif exp_args["targets"] == "first_half":
                meets_criteria = idx < len(model.transformer.h) // 2
            elif exp_args["targets"] == "second_half":
                meets_criteria = idx >= len(model.transformer.h) // 2

            if meets_criteria:
                if exp_args["mode"] == "geglu":
                    m.mlp = GPT2MLPGeGLU(model.config, mlp_mult=exp_args["mlp_mult"])
                elif exp_args["mode"] == "double":
                    m.mlp = GPT2MLPDouble(model.config, mlp_mult=exp_args["mlp_mult"])
                elif exp_args["mode"] == "base":
                    m.mlp = GPT2MLP(model.config, mlp_mult=exp_args["mlp_mult"])
                else:
                    raise ValueError("Invalid mode")

            idx += 1
            


def get_run_name(args, exp_args):
    run_name = str(exp_args["mode"]) + "_" + str(exp_args["targets"]) + "_mm:" + str(exp_args["mlp_mult"]) + "_lr:" + str(args["learning_rate"])
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"

    return args, run_name


extra_args = {
    "mode": "base", # base, geglu, double
    "targets": "all", # all, even, odd, first_half, second_half
    "mlp_mult": 4
}