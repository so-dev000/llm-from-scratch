from torch.nn.utils.rnn import pad_sequence


def collate(batch, pad_id=0):
    src_tensors = [item["src"] for item in batch]
    tgt_tensors = [item["tgt"] for item in batch]
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]

    # padding
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_id)

    # generate mask
    src_mask = src_padded != pad_id
    tgt_mask = tgt_padded != pad_id

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "src_text": src_texts,
        "tgt_text": tgt_texts,
    }
