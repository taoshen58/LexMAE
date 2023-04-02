# marco-only
# whole-word-masking
import json
import math
import os

import torch
from torch.utils.data.dataset import Dataset

from peach.common import dir_exists, file_exists, get_line_offsets, save_json, load_json, \
    load_list_from_file, save_pickle, load_pickle
from peach.text.masked_lm import mlm_input_ids_masking
from transformers import AutoTokenizer


def add_data_hyperparameters(parser):
    parser.add_argument("--data_mlm_prob", type=float, default=0.00)
    parser.add_argument("--data_wwm", action="store_true")
    parser.add_argument("--data_rel_paths", type=str, default="none")
    parser.add_argument("--data_columns", type=str, default="1|2")
    parser.add_argument("--data_keys", type=str, default=None)
    parser.add_argument("--data_lens", type=str, default=None)


class DatasetLongText(Dataset):
    DATA_TYPE_SET = set(["train", ])  # "dev", "test"
    LOAD_TYPE_SET = set(["memory", "disk"])

    def __init__(self, data_type, data_dir, data_rel_paths, load_type, data_args, tokenizer):
        assert data_type in self.DATA_TYPE_SET
        assert load_type in self.LOAD_TYPE_SET

        self.data_type = data_type
        self.data_dir = data_dir
        # self.data_format = data_format
        self.data_rel_paths = data_rel_paths
        self.load_type = load_type
        self.data_args = data_args
        self.tokenizer = tokenizer

        self.data_columns = None if self.data_args.data_columns is None else \
            [int(i) for i in self.data_args.data_columns.split("|")]
        self.data_keys = None if self.data_args.data_keys is None else \
            [i for i in self.data_args.data_keys.split("|")]
        self.data_lens = None if self.data_args.data_lens is None else \
            [int(i) for i in self.data_args.data_lens.split("|")]
        assert self.load_type == "disk"

        data_rel_path2offsets = dict()
        for data_rel_path in self.data_rel_paths.split("|"):
            raw_data_path = os.path.join(self.data_dir, data_rel_path)
            assert file_exists(raw_data_path)
            offsets_path = raw_data_path + ".offset.pkl"
            if file_exists(offsets_path):
                offsets = load_pickle(offsets_path)
            else:
                offsets = get_line_offsets(raw_data_path,)
                save_pickle(offsets, offsets_path, protocol=4)
            data_rel_path2offsets[data_rel_path] = offsets

        data_rel_path_list = list(data_rel_path2offsets.keys())
        data_rel_path_list.sort()

        self.example_list = []
        for data_rel_path in data_rel_path_list:
            for offset in data_rel_path2offsets[data_rel_path]:
                self.example_list.append((data_rel_path, offset,))

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, item):
        rel_path, offset = self.example_list[item]
        with open(os.path.join(self.data_dir, rel_path), encoding="utf-8") as fp:
            fp.seek(offset)
            line = fp.readline()
            if rel_path.endswith("jsonl"):
                data = json.loads(line)
                text_list = [data[key] or "-" for key in self.data_keys]
            else:
                line = line.strip(os.linesep)
                data = line.split("\t")
                text_list = [data[col] or "-" for col in self.data_columns]

        # tokenization
        if len(text_list) <= 2 and self.data_lens is None:
            tkr_outputs = self.tokenizer(
                *text_list, add_special_tokens=True, truncation="longest_first",
                max_length=self.data_args.max_length)
            input_ids = tkr_outputs["input_ids"]
        else:
            assert self.data_lens is not None and len(self.data_lens) == len(text_list)
            input_ids_list = []
            last_idx, acc_len = None, 0
            for idx, (tt, mlen) in enumerate(zip(text_list, self.data_lens)):
                if mlen <= 0:
                    assert last_idx is None
                    last_idx = idx
                    continue
                tkr_outputs = self.tokenizer(
                    tt, add_special_tokens=False, truncation=True, max_length=mlen - 1)  # 1 for [SEP]
                input_ids_list.append(tkr_outputs["input_ids"] + [self.tokenizer.sep_token_id])
                acc_len += len(input_ids_list[-1])
            if last_idx is not None:
                last_len = self.data_args.max_length - acc_len
                assert last_len > 2  # 2 for [CLS] & [SEP]
                tkr_outputs = self.tokenizer(
                    text_list[last_idx], add_special_tokens=False, truncation=True, max_length=last_len - 2)
                input_ids_list.insert(last_idx, tkr_outputs["input_ids"] + [self.tokenizer.sep_token_id])
            input_ids = [self.tokenizer.cls_token_id] + [e for es in input_ids_list for e in es]

        if self.data_args.data_mlm_prob < 1e-5:
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids)}

        # whole word masking
        input_ids = tkr_outputs["input_ids"]
        masked_input_ids, masked_flags, mlm_labels = mlm_input_ids_masking(
            tkr_outputs, self.tokenizer.mask_token_id, self.data_args.data_mlm_prob,
            mlm_rdm_prob=0.1, mlm_keep_prob=0.1, vocab_size=len(self.tokenizer), fix_tgt_num=True,
            whole_word_masking=self.data_args.data_wwm, max_word_len=6, cap_wwm_for_fix=True)

        attention_mask = [1] * len(input_ids)

        # Return
        res_outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "masked_input_ids": masked_input_ids,
            "masked_flags": masked_flags,
            "mlm_labels": mlm_labels,
        }
        return res_outputs








