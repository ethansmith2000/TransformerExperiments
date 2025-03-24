import torch
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, GPT2Block
from types import MethodType
from torch import nn
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention as sdpa


# GPT2Block
def gpt2_block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
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

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    # if use_cache:
    #     outputs = (hidden_states,) + outputs
    # else:
    outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)
    



# GPT2Model
def gpt_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    batch_size = input_ids.shape[0]
    device = input_ids.device


    past_length = 0
    past_key_values = tuple([None] * len(self.h))

    position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)

    position_embeds = self.wpe(position_ids)
    inputs_embeds = self.wte(input_ids)
    hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

    # Attention mask.
    attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    hidden_states = self.drop(hidden_states)

    output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

    use_cache = False

    presents = None
    all_self_attentions = None
    all_cross_attentions = None
    all_hidden_states = None
    for i in range(len(self.h)):
        block, layer_past = self.h[i], past_key_values[i]

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )



# GPT2LMHeadModel
def gpt_lm_head_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Flatten the tokens
        loss = self.loss_function(
            lm_logits,
            labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
        cross_attentions=transformer_outputs.cross_attentions,
    )

class AttentionBase(nn.Module):
    """
    Causal multihead attention that uses torch's SDPA 
    """
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim*3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,):

        b, n, d, h = (*x.shape, self.heads)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (self.to_qkv(x).chunk(3, dim=-1)))
        outputs = (self.to_out(rearrange(sdpa(q, k, v, is_causal=True), 'b h n d -> b n (h d)')), None)
        return outputs


def patch_gpt(model):
    # patch attentions
    for n, m in model.named_modules():
        if hasattr(m, "attn"):
            dim = model.config.n_embd
            heads = m.attn.num_heads

            m.attn = AttentionBase(dim=dim, heads=heads)
        if hasattr(m, "mlp"):
            dim = model.config.n_embd
            inner_dim = model.config.n_inner if model.config.n_inner is not None else 4 * model.config.n_embd
            m.mlp.c_fc = nn.Linear(dim, inner_dim, bias=True)
            m.mlp.c_proj = nn.Linear(inner_dim, dim, bias=True)


    for block in model.transformer.h:
        block.forward = MethodType(gpt2_block_forward, block)
    model.transformer.forward = MethodType(gpt_model_forward, model.transformer)
    model.forward = MethodType(gpt_lm_head_model_forward, model)


    # init all 2d weights xavier normal
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if m.weight.dim() >= 2:
                nn.init.xavier_normal_(m.weight)

    # init all biases to 0, including norm params
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    return model

