import torch

from utils.masking import combine_masks, create_causal_mask

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
BEAM_SIZE = 4
LENGTH_PENALTY = 0.6
MAX_OUTPUT_OFFSET = 50


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def length_penalty(length, alpha):
    return ((5 + length) / 6) ** alpha


def translate_sentence_beam(
    model,
    sentence,
    en_tokenizer,
    ja_tokenizer,
    beam_size=BEAM_SIZE,
    alpha=LENGTH_PENALTY,
    device=None,
):
    if device is None:
        device = get_device()

    model.eval()

    src_ids = en_tokenizer.encode(sentence, add_special_tokens=True)
    max_length = len(src_ids) + MAX_OUTPUT_OFFSET

    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    src_padding_mask = src_tensor != PAD_IDX
    encoder_src_mask = src_padding_mask.unsqueeze(1) & src_padding_mask.unsqueeze(2)

    beams = [
        {
            "tokens": torch.tensor([[BOS_IDX]], device=device),
            "score": 0.0,
            "finished": False,
            "normalized_score": 0.0,
        }
    ]

    for _ in range(max_length):
        all_candidates = []

        for beam in beams:
            # reached EOS
            if beam["finished"]:
                all_candidates.append(beam)
                continue

            # beam["tokens"]: (1, current_length)
            tgt_tensor = beam["tokens"]
            tgt_len = tgt_tensor.size(1)

            causal_mask = create_causal_mask(tgt_len, device=device)

            tgt_padding_mask = tgt_tensor != PAD_IDX

            tgt_combined_mask = combine_masks(tgt_padding_mask, causal_mask)

            with torch.no_grad():
                output = model(
                    src_tensor,
                    tgt_tensor,
                    encoder_src_mask=encoder_src_mask,
                    decoder_src_mask=src_padding_mask,
                    tgt_mask=tgt_combined_mask,
                )

            #  get last output
            # output: (1, current_length, vocab_size)
            logits = output[:, -1, :]  # (1, vocab_size)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # get top-k tokens
            # top_k_log_probs: (1, beam_size)
            # top_k_indices: (1, beam_size)
            top_k_log_probs, top_k_indices = log_probs.topk(beam_size)

            # get new candidates from top-k tokens
            for i in range(beam_size):
                new_token = top_k_indices[0, i].item()
                new_log_prob = top_k_log_probs[0, i].item()

                new_tokens = torch.cat(
                    [tgt_tensor, torch.tensor([[new_token]], device=device)], dim=1
                )

                new_score = beam["score"] + new_log_prob

                is_finished = new_token == EOS_IDX

                current_length = new_tokens.size(1)
                penalty = length_penalty(current_length, alpha)
                normalized_score = new_score / penalty

                candidate = {
                    "tokens": new_tokens,
                    "score": new_score,
                    "finished": is_finished,
                    "normalized_score": normalized_score,
                }

                all_candidates.append(candidate)

        sorted_candidates = sorted(
            all_candidates,
            key=lambda x: x["normalized_score"],
            reverse=True,
        )
        # get next beam (top-k)
        beams = sorted_candidates[:beam_size]

        if all(beam["finished"] for beam in beams):
            break

    best_beam = max(beams, key=lambda b: b["normalized_score"])

    tgt_ids = best_beam["tokens"].squeeze(0).tolist()

    # remove special tokens
    filtered_ids = [id for id in tgt_ids if id not in [PAD_IDX, BOS_IDX, EOS_IDX]]
    return ja_tokenizer.decode(filtered_ids).strip()
