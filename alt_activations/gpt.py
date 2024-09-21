


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


class NewGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, value_act=None, post_attn_act=None, power=1.0):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        # self.value_act = nn.Identity() if value_act is None else Activation(value_act)
        # self.post_attn_act = nn.Identity() if post_attn_act is None else Activation(post_attn_act)
        dim = self.c_attn.nf // 3
        self.value_act = nn.Identity() if value_act is None else LinearAct(dim, dim, activation_type=value_act, power=power, pre_act=True)
        self.post_attn_act = nn.Identity() if post_attn_act is None else LinearAct(dim, dim, activation_type=post_attn_act, power=power, pre_act=True)

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
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        value = self.value_act(value)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self.post_attn_act(attn_output)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


def patch_attn(model, value_act=None, post_attn_act=None, power=1.0):
    conf = model.config
    idx = 0
    for n,m in model.named_modules():
        if hasattr(m, "attn"):
            del m.attn
            m.add_module("attn", NewGPT2Attention(conf, is_cross_attention=False, layer_idx=None, value_act=value_act, post_attn_act=post_attn_act, power=power))
            idx += 1
            print('current idx', idx)


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