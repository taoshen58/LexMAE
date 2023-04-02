import logging

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.models.distilbert.modeling_distilbert import DistilBertForMaskedLM

from peach.nn_utils.general import len_mask, mask_out_cls_sep, add_prefix
from peach.nn_utils.masked_lm import text_part_mask_generation

def add_model_hyperparameters(parser):
    # applicable to all
    parser.add_argument("--keep_special_tokens", action="store_true")
    # disable some part
    parser.add_argument("--disable_partitions", type=int, default=None, help="1-based partition idx")

    return ["keep_special_tokens", "disable_partitions", ]


class SpladePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, token_embeddings, attention_mask, **kwargs):
        # fix 17
        token_embeddings.masked_fill_((attention_mask == 0).unsqueeze(-1), 0.)
        torch.relu_(token_embeddings)
        with torch.no_grad():
            max_indices = torch.argmax(1. + token_embeddings, dim=1).unsqueeze(1).detach()  # [bs,1,V]
        sentence_embedding = torch.gather(token_embeddings, dim=1, index=max_indices).squeeze(1)
        sentence_embedding = torch.log(1. + sentence_embedding)
        return_dict = {
            "sentence_embedding": sentence_embedding,
            "sparse_sentence_embedding": sentence_embedding, }
        return return_dict


class BertSpladeEnocder(BertForMaskedLM):
    def __init__(self, config):
        if not hasattr(config, "keep_special_tokens"):
            config.keep_special_tokens = False
            logging.warning("keep_special_tokens is not given when run SPLADE, set to False by default.")
        if not hasattr(config, "disable_partitions"):
            config.disable_partitions = None
            logging.warning("disable_partitions is not given when run SPLADE, set to None by default.")

        super().__init__(config)
        self.spalde_pooler = SpladePooler(config)
        self.special_token_ids = [101, 102]

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            # head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
            # labels=None, output_attentions=None, output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        encoder_outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            labels=None, output_hidden_states=True) #  token_type_ids, position_ids

        splade_pool_mask = attention_mask
        if (not self.config.keep_special_tokens) or self.config.disable_partitions:
            with torch.no_grad():
                valid_text_mask = text_part_mask_generation(input_ids, self.special_token_ids, attention_mask)
                if not self.config.keep_special_tokens:
                    splade_pool_mask = valid_text_mask
                if self.config.disable_partitions:  # disable_partitions is not None and disable_partitions > 0
                    special_token_mask = (1 - valid_text_mask) * attention_mask
                    num_special_tokens = special_token_mask.sum(-1)
                    if num_special_tokens.shape[0] > 0 and num_special_tokens[0].cpu().item() > 2:  # just apply to doc
                        accu_special_token_mask = torch.cumsum(special_token_mask, -1)  # [bs,sl]
                        enable_part_mask = (accu_special_token_mask != self.config.disable_partitions).to(torch.long)
                        splade_pool_mask = splade_pool_mask * enable_part_mask

        pooling_out_dict = self.spalde_pooler(
            token_embeddings=encoder_outputs.logits, attention_mask=splade_pool_mask,)

        # data to return
        dict_for_return = {
            "hidden_states": encoder_outputs.hidden_states[-1],
            "all_hidden_states": encoder_outputs.hidden_states,
            "prediction_logits": encoder_outputs.logits,
        }
        dict_for_return.update(pooling_out_dict)

        return dict_for_return if return_dict else dict_for_return["sentence_embedding"]


class RobertaSpladeEnocder(RobertaForMaskedLM):
    def __init__(self, config):
        if not hasattr(config, "keep_special_tokens"):
            config.keep_special_tokens = False
            logging.warning("keep_special_tokens is not given when run SPLADE, set to False by default.")
        if not hasattr(config, "disable_partitions"):
            config.disable_partitions = None
            logging.warning("disable_partitions is not given when run SPLADE, set to None by default.")

        super().__init__(config)
        self.spalde_pooler = SpladePooler(config)
        self.special_token_ids = [0, 2,]  # [CLS] [SEP]

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            # head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
            # labels=None, output_attentions=None, output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        encoder_outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=None, output_hidden_states=True) #  token_type_ids, position_ids

        splade_pool_mask = attention_mask
        if (not self.config.keep_special_tokens) or self.config.disable_partitions:
            with torch.no_grad():
                valid_text_mask = text_part_mask_generation(input_ids, self.special_token_ids, attention_mask)
                if not self.config.keep_special_tokens:
                    splade_pool_mask = valid_text_mask
                if self.config.disable_partitions:  # disable_partitions is not None and disable_partitions > 0
                    special_token_mask = (1 - valid_text_mask) * attention_mask
                    num_special_tokens = special_token_mask.sum(-1)
                    if num_special_tokens.shape[0] > 0 and num_special_tokens[0].cpu().item() > 2:  # just apply to doc
                        accu_special_token_mask = torch.cumsum(special_token_mask, -1)  # [bs,sl]
                        enable_part_mask = (accu_special_token_mask != self.config.disable_partitions).to(torch.long)
                        splade_pool_mask = splade_pool_mask * enable_part_mask

        pooling_out_dict = self.spalde_pooler(
            token_embeddings=encoder_outputs.logits, attention_mask=splade_pool_mask,)

        # data to return
        dict_for_return = {
            "hidden_states": encoder_outputs.hidden_states[-1],
            "all_hidden_states": encoder_outputs.hidden_states,
            "prediction_logits": encoder_outputs.logits,
        }
        dict_for_return.update(pooling_out_dict)

        return dict_for_return if return_dict else dict_for_return["sentence_embedding"]

class DistilBertSpladeEnocder(DistilBertForMaskedLM):
    def __init__(self, config):
        if not hasattr(config, "keep_special_tokens"):
            config.keep_special_tokens = False
            logging.warning("keep_special_tokens is not given when run SPLADE, set to False by default.")
        if not hasattr(config, "disable_partitions"):
            config.disable_partitions = None
            logging.warning("disable_partitions is not given when run SPLADE, set to None by default.")

        super().__init__(config)
        self.spalde_pooler = SpladePooler(config)
        self.special_token_ids = [101, 102]

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            attention_mask_3d=None,
            # head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
            # labels=None, output_attentions=None, output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        encoder_outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask_3d if attention_mask_3d is not None else attention_mask,
            labels=None, output_hidden_states=True) #  token_type_ids, position_ids

        splade_pool_mask = attention_mask
        if (not self.config.keep_special_tokens) or self.config.disable_partitions:
            with torch.no_grad():
                valid_text_mask = text_part_mask_generation(input_ids, self.special_token_ids, attention_mask)
                if not self.config.keep_special_tokens:
                    splade_pool_mask = valid_text_mask
                if self.config.disable_partitions:  # disable_partitions is not None and disable_partitions > 0
                    special_token_mask = (1 - valid_text_mask) * attention_mask
                    num_special_tokens = special_token_mask.sum(-1)
                    if num_special_tokens.shape[0] > 0 and num_special_tokens[0].cpu().item() > 2:  # just apply to doc
                        accu_special_token_mask = torch.cumsum(special_token_mask, -1)  # [bs,sl]
                        enable_part_mask = (accu_special_token_mask != self.config.disable_partitions).to(torch.long)
                        splade_pool_mask = splade_pool_mask * enable_part_mask

        pooling_out_dict = self.spalde_pooler(
            token_embeddings=encoder_outputs.logits, attention_mask=splade_pool_mask,)

        # data to return
        dict_for_return = {
            "hidden_states": encoder_outputs.hidden_states[-1],
            "all_hidden_states": encoder_outputs.hidden_states,
            "prediction_logits": encoder_outputs.logits,
        }
        dict_for_return.update(pooling_out_dict)

        return dict_for_return if return_dict else dict_for_return["sentence_embedding"]


class SpladePoolerLargeMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, token_embeddings, attention_mask, **kwargs):
        saturated_token_embeddings = torch.log(1. + torch.relu(token_embeddings))  #[bs,seq,V] [CLS]
        saturated_token_embeddings = saturated_token_embeddings * attention_mask.unsqueeze(-1)
        sentence_embedding = torch.max(saturated_token_embeddings, dim=1).values
        return_dict = {
            # "token_embeddings": token_embeddings,
            # "attention_mask": attention_mask,
            "saturated_token_embeddings": saturated_token_embeddings,
            "sentence_embedding": sentence_embedding,
            "sparse_sentence_embedding": sentence_embedding, }
        return return_dict
