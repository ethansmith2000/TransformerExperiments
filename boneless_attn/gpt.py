import torch
from torch import nn
from typing import Optional, Tuple
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, eager_attention_forward, ALL_ATTENTION_FUNCTIONS


class BaselineAttention(GPT2Attention):
    """
    Same attention class, but patched to let wq, wk, wv to be seperate matrices
    """

    def __init__(self, config, is_cross_attention=False, layer_idx=None, **kwargs):
        super().__init__(config, is_cross_attention, layer_idx)
        del self.c_attn
        del self.c_proj
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_attn = nn.Linear(config.n_embd, config.n_embd, bias=True)

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
        **kwargs,
    ):
        # query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query_states = self.q_attn(hidden_states)
        key_states = self.k_attn(hidden_states)
        value_states = self.v_attn(hidden_states)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None

        is_cross_attention = encoder_hidden_states is not None
        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
            else:
                # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
                # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
                # not necessarily to eager (if mentionned options are provided).
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.o_attn(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs



class CorrelationAttention(BaselineAttention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, **kwargs):
        super().__init__(config, is_cross_attention, layer_idx)
        if kwargs.get("mod_q", False):
            self.q_attn = nn.Identity()
        if kwargs.get("mod_k", False):
            self.k_attn = nn.Identity()

        # bonus 
        if kwargs.get("mod_v", False):
            self.v_attn = nn.Identity()
        if kwargs.get("mod_o", False):
            self.o_attn = nn.Identity()

def init_to_identity(module):
    if isinstance(module, nn.Linear):
        module.weight.data.fill_(0.0)
        module.weight.data[:, :] = torch.eye(module.weight.data.shape[1])

class CorrelationInitAttention(BaselineAttention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, **kwargs):
        super().__init__(config, is_cross_attention, layer_idx)
        if kwargs.get("mod_q", False):
            init_to_identity(self.q_attn)
        if kwargs.get("mod_k", False):
            init_to_identity(self.k_attn)

        # bonus
        if kwargs.get("mod_v", False):
            init_to_identity(self.v_attn)
        if kwargs.get("mod_o", False):
            init_to_identity(self.o_attn)


class ResidualLinear(nn.Linear):
    def forward(self, input):
        return input + super().forward(input)

# class ResidualLinear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=False, scale=1.0, trainable_scale=False):
#         super().__init__(in_features, out_features, bias)
#         scale = torch.Tensor([scale])
#         if trainable_scale:
#             self.scale = nn.Parameter(scale)
#         else:
#             self.register_buffer("scale", scale)

#     def forward(self, input):
#         return input * self.scale + super().forward(input) * (1 - self.scale)


class ResidualAttention(BaselineAttention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, **kwargs):
        super().__init__(config, is_cross_attention, layer_idx)
        if kwargs.get("mod_q", False):
            self.q_attn = ResidualLinear(config.n_embd, config.n_embd, bias=False)
            # nn.init.xavier_uniform_(self.q_attn.weight)
        if kwargs.get("mod_k", False):
            self.k_attn = ResidualLinear(config.n_embd, config.n_embd, bias=False)
            # nn.init.xavier_uniform_(self.k_attn.weight)
        # bonus
        if kwargs.get("mod_v", False):
            self.v_attn = ResidualLinear(config.n_embd, config.n_embd, bias=False)
            # nn.init.xavier_uniform_(self.v_attn.weight)
        if kwargs.get("mod_o", False):
            self.o_attn = ResidualLinear(config.n_embd, config.n_embd, bias=True)
            # nn.init.xavier_uniform_(self.o_attn.weight)


def patch_model(model, optimizer, args, exp_args):
    idx = 0
    for n,m in model.named_modules():
        if hasattr(m, "attn"):
            if exp_args["mode"] == "correlation":
                m.attn = CorrelationAttention(model.config, **exp_args)
            elif exp_args["mode"] == "correlation_init":
                m.attn = CorrelationInitAttention(model.config, **exp_args)
            elif exp_args["mode"] == "residual":
                m.attn = ResidualAttention(model.config, **exp_args)
            elif exp_args["mode"] == "base":
                m.attn = BaselineAttention(model.config, **exp_args)
            elif exp_args["mode"] == "dummy":
                m.attn = nn.Identity()
            else:
                raise ValueError(f"Invalid mode: {exp_args['mode']}")

    return model, optimizer
            


def get_run_name(args, exp_args):
    run_name = "mode_" + exp_args["mode"] + "_modq_" + str(exp_args["mod_q"]) + "_modk_" + str(exp_args["mod_k"]) + "_modv_" + str(exp_args["mod_v"]) + "_ modo_" + str(exp_args["mod_o"]) + "_lr:" + str(args["learning_rate"])
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"

    return args, run_name


extra_args = {
    # "mode": ["correlation", "correlation_init", "residual", "dummy", "base"],
    # "mod_q": [True, False],
    # "mod_k": [True, False],
    # "mod_v": [True, False],
    # "mod_o": [True, False],

    # baseline
    "mode": "base",
    "mod_q": False,
    "mod_k": False,
    "mod_v": False,
    "mod_o": False,

    # correlation, no v/o
    # "mode": "correlation",
    # "mod_q": True,
    # "mod_k": True,
    # "mod_v": False,
    # "mod_o": False,

    # pure correlation
    # "mode": "correlation",
    # "mod_q": True,
    # "mod_k": True,
    # "mod_v": True,
    # "mod_o": True,

    # init correlation no v/o
    # "mode": "correlation_init",
    # "mod_q": True,
    # "mod_k": True,
    # "mod_v": False,
    # "mod_o": False,

    # init correlation
    # "mode": "correlation_init",
    # "mod_q": True,
    # "mod_k": True,
    # "mod_v": True,
    # "mod_o": True,

    # residual no v/o
    # "mode": "residual",
    # "mod_q": True,
    # "mod_k": True,
    # "mod_v": False,
    # "mod_o": False,
    # "trainable_scale": True,
    
    # residual
    # "mode": "residual",
    # "mod_q": True,
    # "mod_k": True,
    # "mod_v": True,
    # "mod_o": True,
    # "trainable_scale": True,
    
    
    
    
    
}
