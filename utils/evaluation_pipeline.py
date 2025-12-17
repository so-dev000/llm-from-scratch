import torch
from tqdm import tqdm


def evaluate_bleu(model, test_dataloader, src_tokenizer, tgt_tokenizer, config):
    try:
        from torchmetrics.text import BLEUScore
    except ImportError:
        raise ImportError("torchmetrics required for BLEU evaluation")

    from utils.inference_pipeline import translate_batch

    model.eval()
    device = next(model.parameters()).device

    bleu = BLEUScore()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating BLEU"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            src_texts = []
            for src_seq in src:
                tokens = src_seq.tolist()
                text = src_tokenizer.decode(tokens, skip_special_tokens=True)
                src_texts.append(text)

            predictions = translate_batch(
                model, src_texts, src_tokenizer, tgt_tokenizer, config, strategy="beam"
            )

            references = []
            for tgt_seq in tgt:
                tokens = tgt_seq.tolist()
                text = tgt_tokenizer.decode(tokens, skip_special_tokens=True)
                references.append(text)

            all_predictions.extend(predictions)
            all_references.extend([[ref] for ref in references])

    bleu_score = bleu(all_predictions, all_references)

    return {"bleu": bleu_score.item()}


def evaluate_perplexity(model, test_dataloader, config):
    raise NotImplementedError()


def evaluate_model(model, test_dataloader, config, **kwargs):
    if config.model.model_type == "transformer":
        src_tokenizer = kwargs.get("src_tokenizer")
        tgt_tokenizer = kwargs.get("tgt_tokenizer")

        if src_tokenizer is None or tgt_tokenizer is None:
            raise ValueError(
                "src_tokenizer and tgt_tokenizer required for Transformer evaluation"
            )

        return evaluate_bleu(
            model, test_dataloader, src_tokenizer, tgt_tokenizer, config
        )

    elif config.model.model_type == "gpt":
        return evaluate_perplexity(model, test_dataloader, config)

    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")
