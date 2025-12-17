from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class DecodingStrategy(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def decode(self, model, src_tokens, src_mask, max_len):
        pass


class BeamSearch(DecodingStrategy):
    def decode(self, model, src_tokens, src_mask, max_len):
        device = src_tokens.device
        batch_size = src_tokens.size(0)
        beam_size = self.config.beam_size
        context = model.prepare_context(src_tokens, src_mask)
        bos_idx = self.config.bos_idx
        eos_idx = self.config.eos_idx

        beams = [
            [(torch.tensor([bos_idx], device=device), 0.0)] for _ in range(batch_size)
        ]

        for _ in range(max_len):
            all_candidates = []
            for batch_idx in range(batch_size):
                candidates = []
                for seq, score in beams[batch_idx]:
                    if seq[-1].item() == eos_idx:
                        candidates.append((seq, score))
                        continue

                    tgt_input = seq.unsqueeze(0)
                    context_subset = (
                        context
                        if context is None
                        else {
                            k: v[batch_idx : batch_idx + 1]
                            if isinstance(v, torch.Tensor)
                            else v
                            for k, v in context.items()
                        }
                    )
                    logits = model.generate_next_token(tgt_input, context_subset)
                    log_probs = F.log_softmax(logits, dim=-1)
                    top_log_probs, top_indices = log_probs.topk(beam_size)

                    for k in range(beam_size):
                        new_seq = torch.cat(
                            [seq, top_indices[0, k].unsqueeze(0)], dim=0
                        )
                        new_score = score + top_log_probs[0, k].item()
                        candidates.append((new_seq, new_score))

                candidates = sorted(
                    candidates,
                    key=lambda x: x[1]
                    / ((5 + len(x[0])) / 6) ** self.config.length_penalty,
                    reverse=True,
                )[:beam_size]
                all_candidates.append(candidates)
            beams = all_candidates
            if all(all(seq[-1].item() == eos_idx for seq, _ in beam) for beam in beams):
                break

        results = []
        for beam in beams:
            best_seq, _ = beam[0]
            results.append(best_seq)
        return results


class GreedyDecoding(DecodingStrategy):
    def decode(self, model, src_tokens, src_mask, max_len):
        device = src_tokens.device
        batch_size = src_tokens.size(0)
        context = model.prepare_context(src_tokens, src_mask)
        bos_idx = self.config.bos_idx
        eos_idx = self.config.eos_idx
        results = []

        for batch_idx in range(batch_size):
            output_tokens = [bos_idx]
            for _ in range(max_len):
                tgt_input = torch.tensor([output_tokens], device=device)
                context_subset = (
                    context
                    if context is None
                    else {
                        k: v[batch_idx : batch_idx + 1]
                        if isinstance(v, torch.Tensor)
                        else v
                        for k, v in context.items()
                    }
                )
                logits = model.generate_next_token(tgt_input, context_subset)
                next_token = logits.argmax(dim=-1).item()
                output_tokens.append(next_token)
                if next_token == eos_idx:
                    break
            results.append(torch.tensor(output_tokens, device=device))
        return results


class SamplingDecoder(DecodingStrategy):
    def decode(self, model, src_tokens, src_mask, max_len):
        raise NotImplementedError()
