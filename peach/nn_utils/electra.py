import torch
from peach.nn_utils.masked_lm import special_token_mask_generation, text_part_mask_generation, prob_mask_generation


def gumbel_noise(tensor, eps=1e-9):
    noise = torch.zeros_like(tensor).uniform_(0, 1)
    return -torch.log(-torch.log(noise + eps) + eps)


def gumbel_sample(logits, temperature=1.0):
    return ((logits / temperature) + gumbel_noise(logits)).argmax(dim=-1)


def electra_token_masking(
        input_ids, attention_mask, mask_token_id, special_token_ids,
        special_token_replace_prob=0.85, mlm_mask_prob=0.15,
        random_replace_prob=0.0, vocab_size=None,
):
    with torch.no_grad():
        mask_text_part = (1 - special_token_mask_generation(input_ids, special_token_ids)) * attention_mask

        # prob_mask_generation
        masked_flags = prob_mask_generation(input_ids, mlm_mask_prob) * mask_text_part

        masked_input_ids = input_ids.clone()
        gen_mlm_labels = input_ids.masked_fill((1-masked_flags).to(torch.bool), -100)

        mask_spt_replace = masked_flags.clone()
        # random masking
        if random_replace_prob > 1e-6:
            random_input_ids = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device)
            mask_rdm_valid_part = (1 - special_token_mask_generation(random_input_ids, special_token_ids))  # in case of special token
            mask_random_replace = prob_mask_generation(input_ids, random_replace_prob) * mask_text_part
            mask_random_replace = mask_random_replace * mask_rdm_valid_part
            masked_input_ids = torch.where(
                mask_random_replace.to(torch.bool), random_input_ids, masked_input_ids)
            mask_spt_replace = mask_spt_replace * (1-mask_random_replace)

        # begin replace special token [MASK]
        mask_spt_replace_prob = prob_mask_generation(input_ids, special_token_replace_prob) * mask_text_part
        masked_input_ids = masked_input_ids.masked_fill(
            (mask_spt_replace * mask_spt_replace_prob).to(torch.bool), mask_token_id)
        return masked_input_ids, mask_text_part, masked_flags, gen_mlm_labels


def electra_token_replacement(
        input_ids, attention_mask,
        mlm_logits, masked_flags, temperature=1.0,
        mask_text_part=None, special_token_ids=None, ):
    with torch.no_grad():
        # generate mask_text_part
        if mask_text_part is None:
            assert special_token_ids is not None
            mask_text_part = text_part_mask_generation(input_ids, special_token_ids, attention_mask)

        # use mask from before to select logits that need sampling
        masked_indices = torch.nonzero(masked_flags, as_tuple=True)
        mlm_tgt_logits = mlm_logits[masked_indices]
        sampled_tokens = gumbel_sample(mlm_tgt_logits, temperature=temperature)

        # scatter the sampled values back to the input
        disc_input_ids = input_ids.clone()
        disc_input_ids[masked_indices] = sampled_tokens

        # generate discriminator labels, with replaced as True and original as False
        disc_bce_labels = (input_ids != disc_input_ids).to(torch.long) * mask_text_part \
                          - 100 * (1-mask_text_part)

        return disc_input_ids, disc_bce_labels





