import torch

from utils.decoding_strategy import BeamSearch, GreedyDecoding


def translate_sentence(
    model, sentence, src_tokenizer, tgt_tokenizer, config, strategy="beam"
):
    results = translate_batch(
        model, [sentence], src_tokenizer, tgt_tokenizer, config, strategy
    )
    return results[0]


def translate_batch(
    model, sentences, src_tokenizer, tgt_tokenizer, config, strategy="beam"
):
    model.eval()
    device = next(model.parameters()).device

    src_ids = []
    for sentence in sentences:
        ids = src_tokenizer.encode(sentence, add_special_tokens=True)
        if len(ids) > config.data.max_length:
            ids = ids[: config.data.max_length - 1] + [
                src_tokenizer.special_tokens["<EOS>"]
            ]
        src_ids.append(ids)

    max_src_len = max(len(ids) for ids in src_ids)
    src_tokens = torch.zeros(len(sentences), max_src_len, dtype=torch.long)
    src_mask = torch.zeros(len(sentences), max_src_len, dtype=torch.bool)

    for i, ids in enumerate(src_ids):
        src_tokens[i, : len(ids)] = torch.tensor(ids)
        src_mask[i, : len(ids)] = True

    src_tokens = src_tokens.to(device)
    src_mask = src_mask.to(device)

    max_output_len = max_src_len + config.inference.max_output_offset

    if strategy == "beam":
        decoder = BeamSearch(config.inference)
    elif strategy == "greedy":
        decoder = GreedyDecoding(config.inference)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    with torch.no_grad():
        output_seqs = decoder.decode(
            model, src_tokens, src_mask, tgt_tokenizer, max_output_len
        )

    translations = []
    for seq in output_seqs:
        tokens = seq.tolist()
        if config.inference.bos_idx in tokens:
            tokens = tokens[1:]
        if config.inference.eos_idx in tokens:
            eos_pos = tokens.index(config.inference.eos_idx)
            tokens = tokens[:eos_pos]

        translation = tgt_tokenizer.decode(tokens, skip_special_tokens=True)
        translations.append(translation)

    return translations


def generate_text(model, prompt, tokenizer, config, strategy="sampling"):
    raise NotImplementedError()


def generate_batch(model, prompts, tokenizer, config, strategy="sampling"):
    raise NotImplementedError()
