import collections
import math
import random


def token_mlm_input_ids_masking(
    input_ids, mask_token_id, special_token_ids,
    mlm_prob=0.15, mlm_rdm_prob=0.1, mlm_keep_prob=0.1, vocab_size=None,
    # fix_tgt_num=False, whole_word_masking=False, max_word_len=None, cap_wwm_for_fix=False
):
    if mlm_prob < 1e-5:  # float safety
        return input_ids, [-100] * len(input_ids), [0] * len(input_ids)

    if not isinstance(special_token_ids, set):
        special_token_ids = set(special_token_ids)

    tgt_token_idxs = [i for i, id in enumerate(input_ids) if id not in special_token_ids]
    random.shuffle(tgt_token_idxs)
    tgt_token_num = math.ceil(len(tgt_token_idxs) * mlm_prob)
    tgt_token_idxs = tgt_token_idxs[:tgt_token_num]

    mlm_mask_prob = 1.0 - mlm_rdm_prob - mlm_keep_prob
    assert mlm_mask_prob >= 0.0

    masked_input_ids, masked_flags, mlm_labels = [], [], []
    for tk_idx, tk_id in enumerate(input_ids):
        if tk_idx in tgt_token_idxs:
            masked_flags.append(1)
            mlm_labels.append(tk_id)

            rdm_prob = random.random()
            if rdm_prob < mlm_mask_prob:
                masked_input_ids.append(mask_token_id)
            elif rdm_prob < (mlm_mask_prob + mlm_rdm_prob):
                masked_input_ids.append(random.randint(0, vocab_size - 1))
            else:
                masked_input_ids.append(tk_id)
        else:
            masked_flags.append(0)
            mlm_labels.append(-100)
            masked_input_ids.append(tk_id)
    return masked_input_ids, masked_flags, mlm_labels


def mlm_input_ids_masking(
        tokenizer_outputs, mask_token_id, mlm_prob=0.15,
        mlm_rdm_prob=0.1, mlm_keep_prob=0.1, vocab_size=None, fix_tgt_num=False,
        whole_word_masking=False, max_word_len=None, cap_wwm_for_fix=False):
    input_ids = tokenizer_outputs["input_ids"]

    if mlm_prob < 1e-5:  # float safety
        return input_ids, [-100] * len(input_ids), [0] * len(input_ids)

    whole_word_ids = tokenizer_outputs.get("whole_word_ids", tokenizer_outputs.word_ids())
    # word_idx2tk_ids = collections.defaultdict(list)
    word_idx2tk_idxs = collections.defaultdict(list)
    all_token_idxs = []

    prev_rel_word_idx, word_idx = -1, -1
    for tk_idx, (tk_id, rel_word_idx) in enumerate(zip(input_ids, whole_word_ids)):
        if rel_word_idx is not None:  # special token
            if rel_word_idx != prev_rel_word_idx:
                word_idx += 1
            # word_idx2tk_ids[word_idx].append(tk_id)
            word_idx2tk_idxs[word_idx].append(tk_idx)
            all_token_idxs.append(tk_idx)
        prev_rel_word_idx = rel_word_idx

    tgt_word_num, tgt_token_num = math.ceil(len(word_idx2tk_idxs) * mlm_prob), math.ceil(len(all_token_idxs) * mlm_prob)

    if whole_word_masking:
        tgt_word_idxs = []
        if fix_tgt_num:
            all_word_idxs = list(word_idx2tk_idxs.keys())
            random.shuffle(all_word_idxs)
            for wi in all_word_idxs:
                if max_word_len is None or len(word_idx2tk_idxs[wi]) <= max_word_len:
                    tgt_word_idxs.append(wi)  # valid words
            if cap_wwm_for_fix:
                tgt_token_idxs = [ti for wi in tgt_word_idxs for ti in word_idx2tk_idxs[wi]][:tgt_token_num]
            else:
                tgt_token_idxs = [ti for wi in tgt_word_idxs[:tgt_word_num] for ti in word_idx2tk_idxs[wi]]
        else:
            for wi in range(len(word_idx2tk_idxs)):
                if max_word_len is None or len(word_idx2tk_idxs[wi]) <= max_word_len:
                    if random.random() < mlm_prob:
                        tgt_word_idxs.append(wi)
            random.shuffle(tgt_word_idxs)
            tgt_token_idxs = [ti for wi in tgt_word_idxs for ti in word_idx2tk_idxs[wi]]
    else:  # token_level masking
        if fix_tgt_num:
            tgt_token_idxs = all_token_idxs.copy()
            random.shuffle(tgt_token_idxs)
            tgt_token_idxs = tgt_token_idxs[:tgt_token_num]
        else:
            tgt_token_idxs = []
            for tidx in all_token_idxs:
                if random.random() < mlm_prob:
                    tgt_token_idxs.append(tidx)
    tgt_token_idxs = set(tgt_token_idxs)

    # begin to mask
    mlm_mask_prob = 1.0 - mlm_rdm_prob - mlm_keep_prob
    assert mlm_mask_prob >= 0.0

    masked_input_ids, masked_flags, mlm_labels = [], [], []
    for tk_idx, tk_id in enumerate(input_ids):
        if tk_idx in tgt_token_idxs:
            masked_flags.append(1)
            mlm_labels.append(tk_id)

            rdm_prob = random.random()
            if rdm_prob < mlm_mask_prob:
                masked_input_ids.append(mask_token_id)
            elif rdm_prob < (mlm_mask_prob + mlm_rdm_prob):
                masked_input_ids.append(random.randint(0, vocab_size - 1))
            else:
                masked_input_ids.append(tk_id)
        else:
            masked_flags.append(0)
            mlm_labels.append(-100)
            masked_input_ids.append(tk_id)

    return masked_input_ids, masked_flags, mlm_labels


# =====================
def deprecated_whole_word_masking(tokenizer_outputs, mask_token_id, vocab_size, mlm_prob=0.15, max_word_len=5, ):
    word_idx2tk_ids = collections.defaultdict(list)
    for tk_idx, (tk_id, word_idx) in enumerate(
            zip(tokenizer_outputs["input_ids"],
                tokenizer_outputs.get("whole_word_ids", tokenizer_outputs.word_ids()))):
        word_idx2tk_ids[word_idx].append(tk_id)

    masked_input_ids, mlm_labels, masked_flags = [], [], []

    if mlm_prob < 1e-5:  # float safety
        input_ids = tokenizer_outputs["input_ids"]
        return input_ids, [-100] * len(input_ids), [0] * len(input_ids)

    for word_idx in range(len(word_idx2tk_ids)):
        tk_ids = word_idx2tk_ids[word_idx]
        rmd_value = random.random()
        if len(tk_ids) > max_word_len or rmd_value > mlm_prob:
            masked_input_ids.extend(tk_ids)
            mlm_labels.extend([-100] * len(tk_ids))
            masked_flags.extend([0] * len(tk_ids))
        else:
            mlm_labels.extend(tk_ids)
            masked_flags.extend([1] * len(tk_ids))
            if rmd_value < mlm_prob * 0.8:
                masked_input_ids.extend([mask_token_id] * len(tk_ids))
            elif rmd_value < mlm_prob * 0.9:
                masked_input_ids.extend(tk_ids)
            else:
                masked_input_ids.extend([random.randint(0, vocab_size - 1) for _ in range(len(tk_ids))])
    return masked_input_ids, mlm_labels, masked_flags