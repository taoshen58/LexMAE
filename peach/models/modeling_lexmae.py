import collections
import copy
import logging
import collections
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead, BertLayer
from peach.nn_utils.masked_lm import text_part_mask_generation
from peach.nn_utils.general import len_mask, masked_pool, exp_mask, zero_mask
from transformers.modeling_outputs import MaskedLMOutput

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, \
    RobertaLayer
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel, \
    gelu, TransformerBlock
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaConfig


def add_model_hyperparameters(parser):
    parser.add_argument("--n_head_layers", type=int, default=2)
    parser.add_argument("--skip_from", type=int, default=None)
    parser.add_argument("--disable_bottleneck", action="store_true")  # ablation
    parser.add_argument("--dec_mlm_bottleneck_src", type=str, default="logits")

    return ["n_head_layers", "skip_from", "disable_bottleneck", "dec_mlm_bottleneck_src"]


def generate_bottleneck_repre(
        input_ids, attention_mask, bottleneck_src,
        special_token_ids=None, word_embeddings_matrix=None,
        last_hidden_states=None, mlm_logits=None,
):
    if bottleneck_src == "cls":
        bottleneck_repre = last_hidden_states[:, 0].contiguous()
    elif bottleneck_src.startswith("logits"):
        with torch.no_grad():
            mask_text_part = text_part_mask_generation(input_ids, special_token_ids, attention_mask)
        pooled_enc_logits = masked_pool(mlm_logits, mask_text_part, high_rank=True, method="max")  # bs,V
        # mlm_logits.masked_fill_((mask_text_part == 0).unsqueeze(-1), 0.)  # apply mask
        # pooled_enc_logits = torch.max(mlm_logits, dim=1).values
        if bottleneck_src == "logits":
            pooled_enc_probs = torch.softmax(pooled_enc_logits, dim=-1)  # bs,V
        elif bottleneck_src == "logits_sat":
            pooled_enc_saturated_logits = torch.log(torch.relu(pooled_enc_logits) + 1.)  # bs,V
            pooled_enc_probs = pooled_enc_saturated_logits / (
                    pooled_enc_saturated_logits.sum(-1, keepdim=True) + 1e-4)
        else:
            raise NotImplementedError(bottleneck_src)
        bottleneck_repre = torch.matmul(pooled_enc_probs, word_embeddings_matrix.detach())    # bs,h
    else:
        raise NotImplementedError(bottleneck_src)
    return bottleneck_repre


class RobertaForEDMLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [
        r"lm_head.decoder.weight", r"lm_head.decoder.bias",
        r"decoder_lm_head.decoder.weight", r"decoder_lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias",
        r"decoder_lm_head.decoder.weight", r"decoder_lm_head.decoder.bias", ]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logging.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # decoder
        self.n_head_layers = getattr(config, "n_head_layers", 2)
        self.skip_from = getattr(config, "skip_from", None)
        self.decoder_heads = nn.ModuleList(
            [RobertaLayer(config) for _ in range(self.n_head_layers)])
        self.decoder_lm_head = RobertaLMHead(config)
        self.update_keys_to_ignore(config, ["decoder_lm_head.decoder.weight"])

        self.init_weights()

        self.special_token_ids = [
            0,  # [CLS]
            2,  # [SEP]
        ]

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def tie_weights(self):
        super().tie_weights()
        self._tie_or_clone_weights(
            self.decoder_lm_head.decoder,
            self.lm_head.decoder)

    def forward_decoder_heads(
            self, cls_rep, dec_input_ids=None, dec_attention_mask=None, dec_token_type_ids=None,
            dec_position_ids=None, dec_inputs_embeds=None, dec_past_key_values_length=0,
            dec_output_attentions=False,
            enc_hidden_states=None,
    ):
        if dec_input_ids is not None:
            input_shape = dec_input_ids.size()
            batch_size, seq_length = input_shape
            device = dec_input_ids.device

            if dec_token_type_ids is None:
                if hasattr(self.roberta.embeddings, "token_type_ids"):
                    buffered_token_type_ids = self.roberta.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                    dec_token_type_ids = buffered_token_type_ids_expanded
                else:
                    dec_token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            dec_embeddings = self.roberta.embeddings(
                input_ids=dec_input_ids,
                position_ids=dec_position_ids,
                token_type_ids=dec_token_type_ids,
                inputs_embeds=dec_inputs_embeds,
                past_key_values_length=dec_past_key_values_length, )
        else:
            assert enc_hidden_states is not None and self.skip_from is not None
            dec_embeddings = enc_hidden_states[self.skip_from]

        attention_mask_etd = self.get_extended_attention_mask(
            dec_attention_mask, dec_attention_mask.shape, dec_attention_mask.device)

        dec_attentions = [] if dec_output_attentions else None
        dec_init_state = torch.cat([cls_rep.unsqueeze(1), dec_embeddings[:, 1:], ], dim=1).contiguous()
        if getattr(self.config, "disable_bottleneck", False):  # for albation
            dec_init_state = dec_embeddings
        dec_hidden_states = [dec_init_state, ]
        for layer in self.decoder_heads:
            layer_out = layer(
                dec_hidden_states[-1],
                attention_mask_etd,
                output_attentions=dec_output_attentions, )
            dec_hidden_states.append(layer_out[0])
            if dec_output_attentions:
                dec_attentions.append(layer_out[1])
        return dec_hidden_states, dec_attentions

    def forward(
        self,
        input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
        head_mask=None, inputs_embeds=None,
        encoder_hidden_states=None, encoder_attention_mask=None,
        dec_input_ids=None, dec_attention_mask=None, dec_token_type_ids=None, dec_position_ids=None,
        enc_cls_rep=None, enc_hidden_states=None,
        enc_mlm_labels=None, dec_mlm_labels=None,
        # labels=None, output_attentions=None, output_hidden_states=None,
        disable_encoding=False, disable_decoding=True,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not disable_encoding:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            prediction_scores = self.lm_head(sequence_output)

            enc_masked_lm_loss = None
            if enc_mlm_labels is not None:
                enc_masked_lm_loss = CrossEntropyLoss()(
                    prediction_scores.view(-1, self.config.vocab_size), enc_mlm_labels.view(-1))

            return_dict = MaskedLMOutput(
                loss=enc_masked_lm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

            assert enc_cls_rep is None and enc_hidden_states is None
            enc_hidden_states = return_dict.hidden_states

            enc_cls_rep = generate_bottleneck_repre(
                input_ids, attention_mask, self.config.dec_mlm_bottleneck_src,
                self.special_token_ids, self.roberta.embeddings.word_embeddings.weight,
                last_hidden_states=enc_hidden_states[-1], mlm_logits=return_dict.logits,
            )
            return_dict["sentence_embedding"] = enc_cls_rep
        else:
            assert enc_cls_rep is not None
            return_dict = collections.OrderedDict()

        if not disable_decoding:
            dec_attention_mask = dec_attention_mask if dec_attention_mask is not None else attention_mask
            dec_hidden_states, dec_attentions = self.forward_decoder_heads(
                enc_cls_rep, dec_input_ids, dec_attention_mask,
                dec_token_type_ids, dec_position_ids,
                dec_output_attentions=True, enc_hidden_states=enc_hidden_states
            )

            dec_logits = self.decoder_lm_head(dec_hidden_states[-1])

            dec_masked_lm_loss = None
            if dec_mlm_labels is not None:
                dec_masked_lm_loss = CrossEntropyLoss()(
                    dec_logits.view(-1, self.config.vocab_size), dec_mlm_labels.view(-1))
            return_dict["dec_loss"] = dec_masked_lm_loss
            return_dict["dec_logits"] = dec_logits
            return_dict["dec_hidden_states"] = dec_hidden_states
            return_dict["dec_attentions"] = dec_attentions

        return return_dict


class XLMRobertaForEDMLM(RobertaForEDMLM):
    config_class = XLMRobertaConfig


class BertForEDMLM(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logging.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # condenser add:
        self.n_head_layers = getattr(config, "n_head_layers", 2)
        self.skip_from = getattr(config, "skip_from", None)
        self.decoder_heads = nn.ModuleList(
            [BertLayer(self.config) for _ in range(self.n_head_layers)])
        self.decoder_cls = BertOnlyMLMHead(config)

        self.init_weights()
        self.special_token_ids = [
            101,  # [CLS]
            102,  # [SEP]
        ]

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def tie_weights(self):
        super().tie_weights()
        # add tie self.decoder_cls <-->
        self._tie_or_clone_weights(
            self.decoder_cls.predictions.decoder,
            self.cls.predictions.decoder)

    def forward_decoder_heads(
            self, cls_rep, dec_input_ids=None, dec_attention_mask=None, dec_token_type_ids=None,
            dec_position_ids=None, dec_inputs_embeds=None, dec_past_key_values_length=0,
            dec_output_attentions=False,
            enc_hidden_states=None,
    ):
        if dec_input_ids is not None:
            dec_embeddings = self.bert.embeddings(
                input_ids=dec_input_ids,
                position_ids=dec_position_ids,
                token_type_ids=dec_token_type_ids,
                inputs_embeds=dec_inputs_embeds,
                past_key_values_length=dec_past_key_values_length,)
        else:
            assert enc_hidden_states is not None and self.skip_from is not None
            dec_embeddings = enc_hidden_states[self.skip_from]
        attention_mask_etd = self.get_extended_attention_mask(
            dec_attention_mask, dec_attention_mask.shape, dec_attention_mask.device)

        dec_attentions = [] if dec_output_attentions else None
        dec_init_state = torch.cat([cls_rep.unsqueeze(1), dec_embeddings[:, 1:],], dim=1).contiguous()
        if getattr(self.config, "disable_bottleneck", False):  # for albation
            dec_init_state = dec_embeddings
        dec_hidden_states = [dec_init_state, ]
        for layer in self.decoder_heads:
            layer_out = layer(
                dec_hidden_states[-1],
                attention_mask_etd,
                output_attentions=dec_output_attentions,)
            dec_hidden_states.append(layer_out[0])
            if dec_output_attentions:
                dec_attentions.append(layer_out[1])
        return dec_hidden_states, dec_attentions

    def forward(
            self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            head_mask=None, inputs_embeds=None,
            encoder_hidden_states=None, encoder_attention_mask=None,
            dec_input_ids=None, dec_attention_mask=None, dec_token_type_ids=None, dec_position_ids=None,
            enc_cls_rep=None, enc_hidden_states=None,
            enc_mlm_labels=None, dec_mlm_labels=None,
            # output_attentions=None, output_hidden_states=None, return_dict=None,
            disable_encoding=False, disable_decoding=True,
            **kwargs):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        # encoding
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not disable_encoding:
            # encoding
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=True,
                output_hidden_states=True,  # force get all hidden states
                return_dict=True,  # force return_dict
            )
            sequence_output = outputs[0]
            prediction_scores = self.cls(sequence_output)

            enc_masked_lm_loss = None
            if enc_mlm_labels is not None:
                # loss 1: middle layer (original)
                enc_masked_lm_loss = CrossEntropyLoss()(
                    prediction_scores.view(-1, self.config.vocab_size), enc_mlm_labels.view(-1))

            return_dict = MaskedLMOutput(
                loss=enc_masked_lm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

            assert enc_cls_rep is None and enc_hidden_states is None
            enc_hidden_states = return_dict.hidden_states
            # enc_cls_rep = return_dict.hidden_states[-1][:, 0].contiguous()
            enc_cls_rep = generate_bottleneck_repre(
                input_ids, attention_mask, self.config.dec_mlm_bottleneck_src,
                self.special_token_ids, self.bert.embeddings.word_embeddings.weight,
                last_hidden_states=enc_hidden_states[-1], mlm_logits=return_dict.logits,
            )
            return_dict["sentence_embedding"] = enc_cls_rep
        else:
            assert enc_cls_rep is not None
            return_dict = collections.OrderedDict()

        if not disable_decoding:
            dec_attention_mask = dec_attention_mask if dec_attention_mask is not None else attention_mask
            dec_hidden_states, dec_attentions = self.forward_decoder_heads(
                enc_cls_rep, dec_input_ids, dec_attention_mask,
                dec_token_type_ids, dec_position_ids,
                dec_output_attentions=True, enc_hidden_states=enc_hidden_states
            )

            dec_logits = self.decoder_cls(dec_hidden_states[-1])
            dec_masked_lm_loss = None
            if dec_mlm_labels is not None:
                dec_masked_lm_loss = CrossEntropyLoss()(
                    dec_logits.view(-1, self.config.vocab_size), dec_mlm_labels.view(-1))
            return_dict["dec_loss"] = dec_masked_lm_loss
            return_dict["dec_logits"] = dec_logits
            return_dict["dec_hidden_states"] = dec_hidden_states
            return_dict["dec_attentions"] = dec_attentions

        return return_dict

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DistilBertForEDMLM(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.n_head_layers = getattr(config, "n_head_layers", 2)
        self.skip_from = getattr(config, "skip_from", None)
        self.decoder_heads = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.n_head_layers)])

        self.decoder_vocab_transform = nn.Linear(config.dim, config.dim)
        self.decoder_vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.decoder_vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.init_weights()

        self.special_token_ids = [
            101,  # [CLS]
            102,  # [SEP]
        ]

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if :obj:`new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (:obj:`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def get_output_embeddings(self):
        return self.vocab_projector

    def set_output_embeddings(self, new_embeddings):
        self.vocab_projector = new_embeddings

    def tie_weights(self):
        super().tie_weights()
        self._tie_or_clone_weights(self.decoder_vocab_projector, self.vocab_projector)

    def cls_head(self, hidden_states):
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        return prediction_logits

    def decoder_cls_head(self, hidden_states):
        prediction_logits = self.decoder_vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.decoder_vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.decoder_vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        return prediction_logits

    def forward_decoder_heads(
            self, cls_rep, dec_input_ids=None, dec_attention_mask=None, dec_token_type_ids=None,
            dec_position_ids=None, dec_inputs_embeds=None, dec_past_key_values_length=0,
            dec_output_attentions=False,
            enc_hidden_states=None,):
        if dec_input_ids is not None:
            dec_embeddings = self.distilbert.embeddings(input_ids=dec_input_ids,)
        else:
            assert enc_hidden_states is not None and self.skip_from is not None
            dec_embeddings = enc_hidden_states[self.skip_from]

        dec_attentions = [] if dec_output_attentions else None
        dec_init_state = torch.cat([cls_rep.unsqueeze(1), dec_embeddings[:, 1:], ], dim=1).contiguous()

        dec_hidden_states = [dec_init_state, ]
        for layer in self.decoder_heads:
            layer_out = layer(
                dec_hidden_states[-1],
                dec_attention_mask,
                output_attentions=dec_output_attentions, )
            dec_hidden_states.append(layer_out[-1])
            if dec_output_attentions:
                assert len(layer_out) == 2
                dec_attentions.append(layer_out[0])
            else:
                assert len(layer_out) == 1
        return dec_hidden_states, dec_attentions

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None, encoder_attention_mask=None,
        dec_input_ids=None, dec_attention_mask=None, dec_token_type_ids=None, dec_position_ids=None,
        enc_cls_rep=None, enc_hidden_states=None,
        enc_mlm_labels=None, dec_mlm_labels=None,
        disable_encoding=False, disable_decoding=True,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not disable_encoding:
            # encoding
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            sequence_output = outputs[0]  # (bs, seq_length, dim)
            prediction_scores = self.cls_head(sequence_output)

            enc_masked_lm_loss = None
            if enc_mlm_labels is not None:
                # loss 1: middle layer (original)
                enc_masked_lm_loss = CrossEntropyLoss()(
                    prediction_scores.view(-1, self.config.vocab_size), enc_mlm_labels.view(-1))

            return_dict = MaskedLMOutput(
                loss=enc_masked_lm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

            assert enc_cls_rep is None and enc_hidden_states is None
            enc_hidden_states = return_dict.hidden_states
            enc_cls_rep = generate_bottleneck_repre(
                input_ids, attention_mask, self.config.dec_mlm_bottleneck_src,
                self.special_token_ids, self.distilbert.embeddings.word_embeddings.weight,
                last_hidden_states=enc_hidden_states[-1], mlm_logits=return_dict.logits,
            )
            return_dict["sentence_embedding"] = enc_cls_rep
        else:
            assert enc_cls_rep is not None
            return_dict = collections.OrderedDict()

        if not disable_decoding:
            dec_attention_mask = dec_attention_mask if dec_attention_mask is not None else attention_mask
            dec_hidden_states, dec_attentions = self.forward_decoder_heads(
                enc_cls_rep, dec_input_ids, dec_attention_mask,
                # dec_token_type_ids, dec_position_ids,
                dec_output_attentions=True, enc_hidden_states=enc_hidden_states
            )
            dec_logits = self.decoder_cls_head(dec_hidden_states[-1])
            dec_masked_lm_loss = None
            if dec_mlm_labels is not None:
                dec_masked_lm_loss = CrossEntropyLoss()(
                    dec_logits.view(-1, self.config.vocab_size), dec_mlm_labels.view(-1))
            return_dict["dec_loss"] = dec_masked_lm_loss
            return_dict["dec_logits"] = dec_logits
            return_dict["dec_hidden_states"] = dec_hidden_states
            return_dict["dec_attentions"] = dec_attentions

        return return_dict
