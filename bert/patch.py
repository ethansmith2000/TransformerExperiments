import torch
import transformers
from torch import nn
import math

from typing import Optional, Tuple

#model.bert.encoder.layer[i].attention.self



class PatchedBertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, dipole_attn=False, second_o=False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

        ##########
        self.dipole_attn = dipole_attn
        self.pos_weights = nn.Parameter(torch.ones(1, self.num_attention_heads, 1, 1))
        self.neg_weights = nn.Parameter(torch.ones(1, self.num_attention_heads, 1, 1))
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        if second_o:
            self.o2 = nn.Linear(config.hidden_size, self.all_head_size)
        else:
            self.o2 = None
        ##########

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            value_2_layer = past_key_value[2]
            attention_mask = encoder_attention_mask
        else:
            ctx = encoder_hidden_states if is_cross_attention else hidden_states
            key_layer = self.transpose_for_scores(self.key(ctx))
            value_layer = self.transpose_for_scores(self.value(ctx))
            value_2_layer = self.transpose_for_scores(self.value2(ctx))
            if is_cross_attention:
                attention_mask = encoder_attention_mask
            elif past_key_value is not None:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
                value_2_layer = torch.cat([past_key_value[2], value_2_layer], dim=2)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer, value_2_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.dipole_attn:
            attn_probs = torch.softmax(attention_scores, dim=-1)
            # attn_probs = self.dropout(attn_probs)
            neg_probs = 1 / (attn_probs + 1e-9)
            if attention_mask is not None:
                attn_mask = attention_mask < -1
                neg_probs = torch.where(attn_mask, torch.zeros_like(neg_probs), neg_probs)
            neg_probs = neg_probs / neg_probs.sum(dim=-1, keepdim=True)

            attn_probs = self.dropout(attn_probs)
            neg_probs = self.dropout(neg_probs)

            pos_context_layer = torch.matmul(attn_probs, value_layer)
            neg_context_layer = torch.matmul(neg_probs, value_2_layer)
            # context_layer = pos_context_layer * self.pos_weights + neg_context_layer * self.neg_weights
            context_layer = pos_context_layer * torch.sigmoid(self.pos_weights) + neg_context_layer * (1-torch.sigmoid(self.pos_weights))
        else:
            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


def patch_model(model, dipole_attn=False, last_n_layers=None):
    for i in range(len(model.bert.encoder.layer)):
        if last_n_layers is not None:
            if i <= len(model.bert.encoder.layer) - last_n_layers - 1:
                continue
        model.bert.encoder.layer[i].attention.self = PatchedBertSelfAttention(model.config, dipole_attn=dipole_attn)
        # also reinit weights of the other parts
        model.bert.encoder.layer[i].attention.output = transformers.models.bert.modeling_bert.BertSelfOutput(model.config)
        model.bert.encoder.layer[i].intermediate = transformers.models.bert.modeling_bert.BertIntermediate(model.config)
        model.bert.encoder.layer[i].output = transformers.models.bert.modeling_bert.BertOutput(model.config)


def gather_params(model, last_n_layers=None):
    params = {}
    if last_n_layers is not None:
        for n, p in model.named_parameters():
            if '.layer.' in n:
                layer_num = int(n.split('.')[3])
                if layer_num >= len(model.bert.encoder.layer) - last_n_layers:
                    params[n] = p
            else:
                if not n in ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight', 'bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.bias']:
                    params[n] = p
    else:
        params = {k: v for k, v in model.named_parameters()}
    return params