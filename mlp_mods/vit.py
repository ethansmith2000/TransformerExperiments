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


# def make_linear(in_dim, out_dim, bias=True, init_type='xavier', **init_kwargs):
#     linear = nn.Linear(in_dim, out_dim, bias=bias)
#     if init_type == 'xavier':
#         nn.init.xavier_uniform_(linear.weight, **init_kwargs)
#     elif init_type == 'kaiming':
#         nn.init.kaiming_uniform_(linear.weight, **init_kwargs)
#     if bias:
#         nn.init.constant_(linear.bias, 0)
#     return linear


# class GeGLU(nn.Module):
#     def __init__(self, in_dim=1024, out_dim=1024, activation_type='gelu', power=1.0, norm=False):
#         super().__init__()
#         self.fc = make_linear(in_dim, out_dim * 2)
#         self.act =  Activation(activation_type=activation_type, power=power)
#         self.norm = torch.nn.LayerNorm(out_dim) if norm else torch.nn.Identity()

#     def forward(self, hidden_states) -> torch.FloatTensor:
#         hidden_states, gate = self.norm(self.fc(hidden_states)).chunk(2, dim=-1)
#         hidden_states = self.act(gate) * hidden_states
#         return hidden_states


# class LinearAct(nn.Module):
#     def __init__(self, in_dim=1024, out_dim=1024, activation_type='gelu', power=1.0, norm=False):
#         super().__init__()
#         self.fc = make_linear(in_dim, out_dim)
#         self.act =  Activation(activation_type=activation_type, power=power)
#         self.norm = torch.nn.LayerNorm(out_dim) if norm else torch.nn.Identity()
    
#     def forward(self, hidden_states) -> torch.FloatTensor:
#         hidden_states = self.act(self.norm(self.fc(hidden_states)))
#         return hidden_states


# class GPT2MLP(nn.Module):
#     """
#     MLP layer but allows for varying number of layers and hidden dimensions and inbetween norms if we want
#     """
#     def __init__(self, 
#                 embed_dim, 
#                 mults=[2,2], 
#                 norms=[False,False],
#                 mode = "base", # geglu
#                 activation_type='gelu', 
#                 power=1.0
#                 ):
#         super().__init__()
#         net = []
#         cur_dim = embed_dim

#         assert len(mults) == len(norms)

#         for i in range(len(mults)):
#             in_dim = embed_dim if i == 0 else cur_dim
#             cur_dim = embed_dim * mults[i]
#             if mode == "geglu":
#                 net.append(GeGLU(in_dim, cur_dim, activation_type, power, norms[i]))
#             else:
#                 net.append(LinearAct(in_dim, cur_dim, activation_type, power, norms[i]))
#         net.append(Conv1D(embed_dim, cur_dim))
#         self.net = nn.Sequential(*net)


#     def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
#         return self.net(hidden_states)



# class GPT2MLPMultipleResidual(nn.Module):

#     """
#     multiple back to back MLPs with residual connections between each
#     """

#     def __init__(self, 
#                 embed_dim, 
#                 mults=[[2,2], [1]], 
#                 interior_norms=[[False,False],[False]],
#                 exterior_norms=[True, True],
#                 mode = ["base", "geglu"], # geglu
#                 activation_type='gelu', 
#                 power=1.0
#                 ):
#         super().__init__()
#         self.sub_mlps = nn.ModuleList()
#         self.norms = nn.ModuleList([
#             nn.LayerNorm(embed_dim) if norm else nn.Identity() for norm in exterior_norms
#         ])
#         for i in range(len(mults)):
#             self.sub_mlps.append(GPT2MLP(embed_dim, mults[i], interior_norms[i], mode[i], activation_type, power))

#     def forward(self, hidden_states) -> torch.FloatTensor:
#         for mlp, norm in zip(self.sub_mlps, self.norms):
#             hidden_states = hidden_states + mlp(norm(hidden_states))
#         return hidden_states




def patch_model(model, args):
    for name, module in model.named_modules():
        if hasattr(module, 'ff'):
            if args['mode'] == "geglu":
                del module.ff
                module.ff = GegluFeedForward(module.ff.net[0].in_features, 
                                                        module.ff.net[0].in_features * args["mult"],
                                                        module.ff.net[2].p,
                                                        # args.activation, 
                                                        # args.act_power
                                                        )
            elif args['mode'] == "double":
                module.ff = DoubleFeedForward(module.ff.net[0].in_features,
                                                        module.ff.net[0].in_features * args["mult"],
                                                        module.ff.net[2].p,
                                                        # args.activation, 
                                                        # args.act_power
                                                        )

    # init weights again
    model.init_weights()


# def get_run_name(args, exp_args):
#     run_name = "mult_" + str(exp_args["mults"]) + "_in_" + str(exp_args["interior_norms"]) + "_en_" + str(exp_args["exterior_norms"]) + "_m_" + str(args["mode"]) + "_t_" + str(args["targets"]) + "_lr:" + str(args["learning_rate"])
#     args["output_dir"] = f"{args['base_output_dir']}/{run_name}"

#     return args, run_name


def get_run_name(args, exp_args):
    run_name = exp_args["mode"] + "_mult_" + str(exp_args["mult"]) + "_t_" + str(exp_args["targets"]) + "_lr:" + str(args["lr"])
    # args["output_dir"] = f"{args['base_output_dir']}/{run_name}"

    return args, run_name


extra_args = {
    # "mults": [[2,2]],
    # "interior_norms": [[False,False]],
    # "exterior_norms": [True],
    # "mode": ["base"],
    # "targets": "all", # all, even, odd, first_half, second_half

    "targets": "all", # all, even, odd, first_half, second_half
    "mode": "double", # base, geglu, double
    "mult": 2,

}