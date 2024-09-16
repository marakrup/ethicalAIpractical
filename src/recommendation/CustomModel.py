from transformers import BertForSequenceClassification, BertModel
from typing import Dict, List, Optional, Tuple, Union
import math
from torch import nn
import torch

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss

from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertOutput,
    BertAttention,
    BertSelfOutput,
    BertSelfAttention,
    BertLayer,
    apply_chunking_to_forward
)

class BertForSequenceClassificationAdapters(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(1, config.num_labels)

        self.adapter_size = config.hidden_size * 2# + config.intermediate_size
        self.embedding_size = config.embedding_size

        self.user_embeddings = nn.Embedding(config.num_users, self.embedding_size)
        #add additional projection layer if adapter_size != self.embedding_size
        if self.adapter_size != self.embedding_size:
          self.user_projection = nn.Linear(self.embedding_size, self.adapter_size)
        self.user_dropout = torch.nn.Dropout(config.user_dropout_prob)
        self.user_norm = torch.nn.LayerNorm(self.embedding_size, eps=config.layer_norm_eps)
        self.adapter_layer = BertLayerAdapters(config)

        self.sample_camparators = torch.nn.parameter.Parameter(data=torch.zeros(config.num_classification_heads,config.hidden_size), requires_grad=True)
        self.sample_camparators.data.normal_(mean=0.0, std=config.initializer_range)
        self.sample_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.register_buffer("adapter_bias", torch.ones(1, self.adapter_size))

        # Initialize weights and apply final processing
        self.post_init()

    def calculate_contrastive_loss(self, current_user, positive_indices, negative_indices, margin: float = 0.5):
        """
        Computes Contrastive Loss
        """

        positive_samples = torch.cartesian_prod(current_user, positive_indices.squeeze(dim=0))
        negative_samples = torch.cartesian_prod(current_user, negative_indices.squeeze(dim=0))


        #similar labesl are zeros / disimilar labels are ones
        positive_samples_labels = torch.zeros(negative_samples.shape[0], device=negative_samples.device)
        negative_samples_labels = torch.ones(positive_samples.shape[0], device=positive_samples.device)

        all_contrastive_samples = torch.concat([positive_samples, negative_samples],dim=0)
        label  = torch.concat([positive_samples_labels, negative_samples_labels],dim=0)

        x1 = self.user_embeddings(all_contrastive_samples[:,0])
        x2 = self.user_embeddings(all_contrastive_samples[:,1])

        x1 = torch.nn.functional.normalize(x1, p=2, dim=-1)
        x2 = torch.nn.functional.normalize(x2, p=2, dim=-1)

        dist = torch.nn.functional.pairwise_distance(x1, x2)

        loss = (1 - label) * torch.pow(dist, 2) + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        loss = torch.mean(loss)

        return loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        users: Optional[torch.Tensor] = None,
        positive_samples: Optional[torch.Tensor] = None,
        negative_samples: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if len(input_ids.shape) > 2:
          input_ids = input_ids.squeeze(dim=0)
          attention_mask = attention_mask.squeeze(dim=0)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        unpooled_output = outputs[0]

        adapter_values = self.adapter_bias

        if users is not None:
          users = users.squeeze(dim=0)
          user_embeds = self.user_embeddings(users)
          user_embeds = self.user_norm(user_embeds)
          user_embeds = self.user_dropout(user_embeds)
          if self.adapter_size != self.embedding_size:
            adapter_values = self.adapter_bias + self.user_projection(user_embeds)
          else:
            adapter_values = self.adapter_bias + user_embeds

        extended_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.size())

        unpooled_output = self.adapter_layer(
            unpooled_output,
            extended_mask,
            output_attentions=output_attentions,
            user_embeds=adapter_values.unsqueeze(dim=1),
        )

        if outputs.hidden_states is not None:
          outputs.hidden_states = outputs.hidden_states + (unpooled_output[0],)
        if outputs.attentions is not None:
          outputs.attentions = outputs.attentions + (unpooled_output[1],)

        pooled_output = self.bert.pooler(unpooled_output[0])

        pooled_output = self.dropout(pooled_output)

        samples = self.sample_norm(self.sample_camparators)
        samples = self.dropout(samples)

        logits = torch.inner(pooled_output, samples)

        click_prediction = self.classifier(logits.detach())

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                if input_ids.shape[0] > 1:
                  loss = loss_fct(logits.view(-1, 5), labels)

                  click_prediction_labels = torch.zeros(5, dtype=torch.long, device=click_prediction.device)
                  click_prediction_labels[labels] = torch.tensor(1, device=click_prediction.device)

                  loss_fct = CrossEntropyLoss(weight=torch.tensor([0.25,1.], device=click_prediction.device))
                  loss += loss_fct(click_prediction.view(-1, 2), click_prediction_labels)
                else:

                  loss = loss_fct(click_prediction.view(-1, 2), labels)

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)


            if positive_samples is not None and negative_samples is not None:
              contrastive_loss = self.calculate_contrastive_loss(users, positive_samples, negative_samples)
              loss += contrastive_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=click_prediction,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertLayerAdapters(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttentionAdapters(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        user_embeds: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        #added
        multiply_keys = user_embeds[:,:,:self.config.hidden_size]
        multiply_values = user_embeds[:,:,self.config.hidden_size:self.config.hidden_size*2]
        multiply_intermediate = None#user_embeds[:,:,self.config.hidden_size*2:]

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            multiply_keys=multiply_keys,
            multiply_values=multiply_values
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, (attention_output, multiply_intermediate)
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        attention_output, multiply_intermediate = attention_output
        intermediate_output = self.intermediate(attention_output)
        #intermediate_output = intermediate_output * multiply_intermediate #added
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertAttentionAdapters(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.self = BertSelfAttentionAdapters(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        multiply_keys: Optional[torch.FloatTensor] = None,
        multiply_values: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            multiply_keys,
            multiply_values,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertSelfAttentionAdapters(BertSelfAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        multiply_keys: Optional[torch.FloatTensor] = None,
        multiply_values: Optional[torch.FloatTensor] = None,
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
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states) * multiply_keys)
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states) * multiply_values)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states) * multiply_keys)
            value_layer = self.transpose_for_scores(self.value(hidden_states) * multiply_values)
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states) * multiply_keys)
            value_layer = self.transpose_for_scores(self.value(hidden_states) * multiply_values)

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
            past_key_value = (key_layer, value_layer)

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

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


from transformers.configuration_utils import PretrainedConfig

class BertConfigAdapters(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        num_classification_heads=2,
        embedding_size=32,
        num_users = None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.num_classification_heads = num_classification_heads
        self.embedding_size = embedding_size
        self.num_users = num_users
