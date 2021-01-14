# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
# Modified in Recurrent VLN-BERT, 2020, Yicong.Hong@anu.edu.au

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.pytorch_transformers.modeling_bert import (BertEmbeddings,
        BertSelfAttention, BertAttention, BertEncoder, BertLayer,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertPooler, BertLayerNorm, BertPreTrainedModel,
		BertPredictionHeadTransform)

logger = logging.getLogger(__name__)

class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)
        self.config = config

    def forward(self, mode, hidden_states, attention_mask, head_mask=None,
            history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        if mode == 'visual':
            mixed_query_layer = mixed_query_layer[:, [0]+list(range(-self.config.directions, 0)), :]

        ''' language feature only provide Keys and Values '''
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_scores)

        return outputs


class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.config = config

    def forward(self, mode, input_tensor, attention_mask, head_mask=None,
            history_state=None):
        ''' transformer processing '''
        self_outputs = self.self(mode, input_tensor, attention_mask, head_mask, history_state)

        ''' feed-forward network with residule '''
        if mode == 'visual':
            attention_output = self.output(self_outputs[0], input_tensor[:, [0]+list(range(-self.config.directions, 0)), :])
        if mode == 'language':
            attention_output = self.output(self_outputs[0], input_tensor)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, mode, hidden_states, attention_mask, head_mask=None,
                history_state=None):

        attention_outputs = self.attention(mode, hidden_states, attention_mask,
                head_mask, history_state)

        ''' feed-forward network with residule '''
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]

        return outputs


class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        # 12 Bert layers
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.config = config

    def forward(self, mode, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):

        if mode == 'visual':
            for i, layer_module in enumerate(self.layer):
                history_state = None if encoder_history_states is None else encoder_history_states[i]

                layer_outputs = layer_module(mode,
                        hidden_states, attention_mask, head_mask[i],
                        history_state)

                concat_layer_outputs = torch.cat((layer_outputs[0][:,0:1,:], hidden_states[:,1:-self.config.directions,:], layer_outputs[0][:,1:self.config.directions+1,:]), 1)
                hidden_states = concat_layer_outputs

                if i == self.config.num_hidden_layers - 1:
                    state_attention_score = layer_outputs[1][:, :, 0, :]
                    lang_attention_score = layer_outputs[1][:, :, -self.config.directions:, 1:-self.config.directions]
                    vis_attention_score = layer_outputs[1][:, :, :, :]

            outputs = (hidden_states, state_attention_score, lang_attention_score, vis_attention_score)

        elif mode == 'language':
            for i, layer_module in enumerate(self.layer):
                history_state = None if encoder_history_states is None else encoder_history_states[i] # default None

                layer_outputs = layer_module(mode,
                        hidden_states, attention_mask, head_mask[i],
                        history_state)
                hidden_states = layer_outputs[0]

                if i == self.config.num_hidden_layers - 1:
                    slang_attention_score = layer_outputs[1]

            outputs = (hidden_states, slang_attention_score)

        return outputs


class BertImgModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """
    def __init__(self, config):
        super(BertImgModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))

        self.apply(self.init_weights)

    def forward(self, mode, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, img_feats=None):

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        if mode == 'visual':
            language_features = input_ids
            concat_embedding_output = torch.cat((language_features, img_feats), 1)
        elif mode == 'language':
            embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                    token_type_ids=token_type_ids)
            concat_embedding_output = embedding_output

        ''' pass to the Transformer layers '''
        encoder_outputs = self.encoder(mode, concat_embedding_output,
                extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) # We "pool" the model by simply taking the hidden state corresponding to the first token

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        return outputs


class VLNBert(BertPreTrainedModel):
    """
    Modified from BertForMultipleChoice to support oscar training.
    """
    def __init__(self, config):
        super(VLNBert, self).__init__(config)
        self.config = config
        self.bert = BertImgModel(config)

        self.vis_lang_LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.state_proj = nn.Linear(config.hidden_size*2, config.hidden_size, bias=True)
        self.state_LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_weights)

    def forward(self, mode, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, img_feats=None):

        outputs = self.bert(mode, input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask, img_feats=img_feats)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        pooled_output = outputs[1]

        if mode == 'language':
            return sequence_output

        elif mode == 'visual':
            # attention scores with respect to agent's state
            language_attentions = outputs[2][:, :, 1:-self.config.directions]
            visual_attentions = outputs[2][:, :, -self.config.directions:]

            language_attention_scores = language_attentions.mean(dim=1)  # mean over the 12 heads
            visual_attention_scores = visual_attentions.mean(dim=1)

            # weighted_feat
            language_attention_probs = nn.Softmax(dim=-1)(language_attention_scores.clone()).unsqueeze(-1)
            visual_attention_probs = nn.Softmax(dim=-1)(visual_attention_scores.clone()).unsqueeze(-1)

            language_seq = sequence_output[:, 1:-self.config.directions, :]
            visual_seq = sequence_output[:, -self.config.directions:, :]

            # residual weighting, final attention to weight the raw inputs
            attended_language = (language_attention_probs * input_ids[:, 1:, :]).sum(1)
            attended_visual = (visual_attention_probs * img_feats).sum(1)

            # update agent's state, unify history, language and vision by elementwise product
            vis_lang_feat = self.vis_lang_LayerNorm(attended_language * attended_visual)
            state_output = torch.cat((pooled_output, vis_lang_feat), dim=-1)
            state_proj = self.state_proj(state_output)
            state_proj = self.state_LayerNorm(state_proj)

            return state_proj, visual_attention_scores
