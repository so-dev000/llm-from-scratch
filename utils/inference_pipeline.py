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
        output_seqs = decoder.decode(model, src_tokens, src_mask, max_output_len)

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


def generate_text(model, prompt, tokenizer, config, strategy="beam"):
    results = generate_batch(model, [prompt], tokenizer, config, strategy)
    return results[0]


def generate_batch(model, prompts, tokenizer, config, strategy="beam"):
    model.eval()
    device = next(model.parameters()).device

    prompt_ids = []
    for prompt in prompts:
        encoding = tokenizer.encode(prompt)
        ids = encoding.ids
        if len(ids) > config.data.max_length:
            ids = ids[: config.data.max_length]
        prompt_ids.append(ids)

    max_prompt_len = max(len(ids) for ids in prompt_ids)
    prompt_tokens = torch.zeros(len(prompts), max_prompt_len, dtype=torch.long)

    for i, ids in enumerate(prompt_ids):
        prompt_tokens[i, : len(ids)] = torch.tensor(ids)

    prompt_tokens = prompt_tokens.to(device)
    max_output_len = config.inference.max_gen_len

    if strategy == "beam":
        decoder = BeamSearch(config.inference)
    elif strategy == "greedy":
        decoder = GreedyDecoding(config.inference)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    with torch.no_grad():
        output_seqs = decoder.decode(model, prompt_tokens, None, max_output_len)

    generations = []
    for i, seq in enumerate(output_seqs):
        tokens = seq.tolist()
        prompt_len = len(prompt_ids[i])
        generated_tokens = tokens[prompt_len:]

        if config.inference.eos_idx in generated_tokens:
            eos_pos = generated_tokens.index(config.inference.eos_idx)
            generated_tokens = generated_tokens[:eos_pos]

        generation = tokenizer.decode(generated_tokens)
        generations.append(generation)

    return generations
