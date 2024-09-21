


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
    def __init__(self, config, activation_type='gelu', power=1.0):
        super().__init__()
        embed_dim = config.hidden_size
        intermediate_size = embed_dim * 4
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


def patch_mlp(model, activation_type, activation_powers):
    idx = 0
    for n,m in model.named_modules():
        if hasattr(m, "mlp"):
            del m.mlp
            m.add_module("mlp", GPT2MLP(model.config, activation_type=activation_type[idx], power=activation_powers[idx]))
            idx += 1
            


def run_name():
    base_str = "base"
    if args["value_act"] is not None:
        base_str = f"{base_str}_vact_{args['value_act']}"
    if args["post_attn_act"] is not None:
        base_str = f"{base_str}_pact_{args['post_attn_act']}"
    args["output_dir"] = f"{args['base_output_dir']}/{base_str}"

    unique_activations = list(set(args['activations']))
    non_gelu = [a for a in unique_activations if a != "gelu"]
    if len(non_gelu) > 0:
        non_gelu = non_gelu[0]
        indices = tuple([i+1 for i, a in enumerate(args['activations']) if a == non_gelu])
        base_str = base_str + "_{}-{}".format(non_gelu, indices)

    unique_powers = list(set(args['activation_powers']))
    non_one = [p for p in unique_powers if p != 1]
    if len(non_one) > 0:
        non_one = non_one[0]
        indices = tuple([i+1 for i, p in enumerate(args['activation_powers']) if p == non_one])
        base_str = base_str + "_{}-{}".format(non_one, indices)


extra_args = {

        "activations": ["gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu", "gelu"],
    "activation_powers": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "value_act": "leaky",
    "post_attn_act": None,
    "attn_power": 3.0,
}