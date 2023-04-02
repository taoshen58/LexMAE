import torch
from functools import reduce


# prob_mask_like
def prob_mask_generation(tensor, prob, dtype=torch.float):
    if prob < 1e-5:
        mask = torch.zeros_like(tensor)
    else:
        mask = (torch.zeros_like(tensor).to(dtype).uniform_(0, 1) < prob).to(torch.long)
    return mask


# mask_with_tokens
def special_token_mask_generation(input_ids, special_token_ids):
    init_no_mask = torch.full_like(input_ids, False, dtype=torch.bool)
    mask_bl = reduce(lambda acc, el: acc | (input_ids == el),
                     special_token_ids, init_no_mask)
    return mask_bl.to(torch.long)


def text_part_mask_generation(input_ids, special_token_ids, attention_mask):
    mask_text_part = (1 - special_token_mask_generation(input_ids, special_token_ids)) * attention_mask
    return mask_text_part


def mlm_input_ids_masking_onthefly(
        input_ids, attention_mask, mask_token_id, special_token_ids,
        mlm_prob=0.15, mlm_rdm_prob=0.1, mlm_keep_prob=0.1,
        vocab_size=None, external_rdm_probs=None,
        extra_masked_flags=None, exclude_flags=None,
        resample_nonmask=False,
):
    with torch.no_grad():
        if external_rdm_probs is None:
            rdm_probs = torch.zeros_like(input_ids).to(torch.float).uniform_(0, 1)
        else:
            rdm_probs = external_rdm_probs
        mask_text_part = text_part_mask_generation(input_ids, special_token_ids, attention_mask)

        if mlm_prob is not None:
            masked_flags = (rdm_probs < mlm_prob).to(torch.long) * mask_text_part
            if extra_masked_flags is not None:
                masked_flags = ((masked_flags + extra_masked_flags * mask_text_part) > 0).to(torch.long)
            if exclude_flags is not None:
                masked_flags = masked_flags * (1 - exclude_flags)
        else:
            assert extra_masked_flags is not None
            masked_flags = extra_masked_flags * mask_text_part
            if exclude_flags is not None:
                masked_flags = masked_flags * (1 - exclude_flags)

        masked_flags_bl = masked_flags.to(torch.bool)
        # label
        mlm_labels = input_ids.clone()
        mlm_labels.masked_fill_(~masked_flags_bl, -100)

        # masked_input_ids
        masked_input_ids = input_ids.clone()
        mlm_mask_prob = 1.0 - mlm_rdm_prob - mlm_keep_prob
        assert mlm_mask_prob >= 0.0

        if resample_nonmask:
            # to ensure randomness
            split_probs = torch.zeros_like(input_ids).to(torch.float).uniform_(0, 1)
            if mlm_mask_prob > 1e-5:
                bl_mask_replace = (split_probs < mlm_mask_prob) & masked_flags_bl
                masked_input_ids.masked_fill_(bl_mask_replace, mask_token_id)
            if mlm_rdm_prob > 1e-5:
                bl_rdm_replace = (split_probs >= mlm_mask_prob) & \
                                 (split_probs < (mlm_mask_prob + mlm_rdm_prob)) & masked_flags_bl
                random_input_ids = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device)
                masked_input_ids = torch.where(bl_rdm_replace, random_input_ids, masked_input_ids)
        else:
            if mlm_mask_prob > 1e-5:
                bl_mask_replace = (rdm_probs < mlm_prob * mlm_mask_prob) & masked_flags_bl
                masked_input_ids.masked_fill_(bl_mask_replace, mask_token_id)
            if mlm_rdm_prob > 1e-5:
                bl_rdm_replace = (rdm_probs >= mlm_prob * mlm_mask_prob) & \
                                 (rdm_probs < mlm_prob * (mlm_mask_prob + mlm_rdm_prob)) & masked_flags_bl
                random_input_ids = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device)
                masked_input_ids = torch.where(bl_rdm_replace, random_input_ids, masked_input_ids)

        return masked_input_ids, masked_flags, mlm_labels, rdm_probs



