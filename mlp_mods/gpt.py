import torch
from torch import nn
from typing import Optional, Tuple
from common.activations import Activation
from types import MethodType


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


class GeGLU(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, activation_type='gelu', power=1.0, norm=False):
        super().__init__()
        self.fc = Conv1D(out_dim * 2, in_dim)
        self.act =  Activation(activation_type=activation_type, power=power)
        self.norm = torch.nn.LayerNorm(out_dim) if norm else torch.nn.Identity()

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states, gate = self.norm(self.fc(hidden_states)).chunk(2, dim=-1)
        hidden_states = self.act(gate) * hidden_states
        return hidden_states


class LinearAct(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, activation_type='gelu', power=1.0, norm=False):
        super().__init__()
        self.fc = Conv1D(out_dim, in_dim)
        self.act =  Activation(activation_type=activation_type, power=power)
        self.norm = torch.nn.LayerNorm(out_dim) if norm else torch.nn.Identity()
    
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.act(self.norm(self.fc(hidden_states)))
        return hidden_states


class GPT2MLP(nn.Module):
    """
    MLP layer but allows for varying number of layers and hidden dimensions and inbetween norms if we want
    """
    def __init__(self, 
                embed_dim, 
                mults=[2,2], 
                norms=[False,False],
                mode = "base", # geglu
                activation_type='gelu', 
                power=1.0
                ):
        super().__init__()
        net = []
        cur_dim = embed_dim

        assert len(mults) == len(norms)

        for i in range(len(mults)):
            in_dim = embed_dim if i == 0 else cur_dim
            cur_dim = embed_dim * mults[i]
            if mode == "geglu":
                net.append(GeGLU(in_dim, cur_dim, activation_type, power, norms[i]))
            else:
                net.append(LinearAct(in_dim, cur_dim, activation_type, power, norms[i]))
        net.append(Conv1D(embed_dim, cur_dim))
        self.net = nn.Sequential(*net)


    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        return self.net(hidden_states)



class GPT2MLPMultipleResidual(nn.Module):

    """
    multiple back to back MLPs with residual connections between each
    """

    def __init__(self, 
                embed_dim, 
                mults=[[2,2], [1]], 
                interior_norms=[[False,False],[False]],
                exterior_norms=[True, True],
                mode = ["base", "geglu"], # geglu
                activation_type='gelu', 
                power=1.0
                ):
        super().__init__()
        self.sub_mlps = nn.ModuleList()
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) if norm else nn.Identity() for norm in exterior_norms
        ])
        for i in range(len(mults)):
            self.sub_mlps.append(GPT2MLP(embed_dim, mults[i], interior_norms[i], mode[i], activation_type, power))

    def forward(self, hidden_states) -> torch.FloatTensor:
        for mlp, norm in zip(self.sub_mlps, self.norms):
            hidden_states = hidden_states + mlp(norm(hidden_states))
        return hidden_states

    


def forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]
    # residual connection
    hidden_states = attn_output + residual

    if encoder_hidden_states is not None:
        # add one self-attention block for cross-attention
        if not hasattr(self, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                "cross-attention layers by setting `config.add_cross_attention=True`"
            )
        residual = hidden_states
        hidden_states = self.ln_cross_attn(hidden_states)
        cross_attn_outputs = self.crossattention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = cross_attn_outputs[0]
        # residual connection
        hidden_states = residual + attn_output
        outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

    # residual = hidden_states
    # hidden_states = self.ln_2(hidden_states)
    # feed_forward_hidden_states = self.mlp(hidden_states)
    # # residual connection
    # hidden_states = residual + feed_forward_hidden_states

    ####
    hidden_states = self.mlp(hidden_states)
    ####

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)



def patch_model(model, optimizer, args, exp_args):
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
                m.mlp = GPT2MLPMultipleResidual(
                    embed_dim = model.config.hidden_size,
                    mults = exp_args["mults"],
                    interior_norms = exp_args["interior_norms"],
                    exterior_norms = exp_args["exterior_norms"],
                    mode = exp_args["mode"]
                )
                m.forward = MethodType(forward, m)

            idx += 1
    return model, optimizer        


def get_run_name(args, exp_args):
    run_name = "mult_" + str(exp_args["mults"]) + "_in_" + str(exp_args["interior_norms"]) + "_en_" + str(exp_args["exterior_norms"]) + "_m_" + str(exp_args["mode"]) + "_t_" + str(exp_args["targets"]) + "_lr:" + str(args["learning_rate"])
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"

    return args, run_name


extra_args = {
    "mults": [[4]],
    "interior_norms": [[False]],
    "exterior_norms": [True],
    "mode": ["base"],
    "targets": "all", # all, even, odd, first_half, second_half


    # "mults": [[4]],
    # "interior_norms": [[False]],
    # "exterior_norms": [True],
    # "mode": ["geglu"],
    # "targets": "all", # all, even, odd, first_half, second_half


    # "mults": [[2,2]],
    # "interior_norms": [[False, False]],
    # "exterior_norms": [True],
    # "mode": ["base"],
    # "targets": "all", # all, even, odd, first_half, second_half

    # "mults": [[2,2]],
    # "interior_norms": [[False, True]],
    # "exterior_norms": [True],
    # "mode": ["base"],
    # "targets": "all", # all, even, odd, first_half, second_half

    # "mults": [[1,1,1,1]],
    # "interior_norms": [[False, False, False, False]],
    # "exterior_norms": [True],
    # "mode": ["base"],
    # "targets": "all", # all, even, odd, first_half, second_half

    # "mults": [[1,1,1,1]],
    # "interior_norms": [[False, True, False, True]],
    # "exterior_norms": [True],
    # "mode": ["base"],
    # "targets": "all", # all, even, odd, first_half, second_half


    # "mults": [[1,1,1,1]],
    # "interior_norms": [[False, False, False, False]],
    # "exterior_norms": [True],
    # "mode": ["geglu"],
    # "targets": "all", # all, even, odd, first_half, second_half

    # "mults": [[1,1,1,1]],
    # "interior_norms": [[False, True, False, True]],
    # "exterior_norms": [True],
    # "mode": ["geglu"],
    # "targets": "all", # all, even, odd, first_half, second_half


    # "mults": [[1,1,1,1]],
    # "interior_norms": [[False, False, False, False]],
    # "exterior_norms": [True],
    # "mode": ["geglu"],
    # "targets": "all", # all, even, odd, first_half, second_half

    # "mults": [[1,1,1,1]],
    # "interior_norms": [[False, True, False, True]],
    # "exterior_norms": [True],
    # "mode": ["geglu"],
    # "targets": "all", # all, even, odd, first_half, second_half

    # "mults": [[2],[2],[2]],
    # "interior_norms": [[False],[False],[False]],
    # "exterior_norms": [True, True, True],
    # "mode": ["base", "base", "base"],
    # "targets": "all", # all, even, odd, first_half, second_half

    # "mults": [[1],[1],[1]],
    # "interior_norms": [[False],[False],[False]],
    # "exterior_norms": [True, True, True],
    # "mode": ["base", "base", "base"],
    # "targets": "all", # all, even, odd, first_half, second_half

}
