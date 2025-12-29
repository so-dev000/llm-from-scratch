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

        has_prompt = src_tokens.size(1) > 0 and (src_tokens != 0).any()

        if has_prompt:
            beams = [[(src_tokens[i], 0.0)] for i in range(batch_size)]
        else:
            beams = [
                [(torch.tensor([bos_idx], device=device), 0.0)]
                for _ in range(batch_size)
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

        has_prompt = src_tokens.size(1) > 0 and (src_tokens != 0).any()

        for batch_idx in range(batch_size):
            if has_prompt:
                output_tokens = src_tokens[batch_idx].tolist()
            else:
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
        device = src_tokens.device
        batch_size = src_tokens.size(0)
        context = model.prepare_context(src_tokens, src_mask)
        bos_idx = self.config.bos_idx
        eos_idx = self.config.eos_idx
        temperature = self.config.temperature
        top_k = self.config.top_k
        top_p = self.config.top_p
        results = []

        has_prompt = src_tokens.size(1) > 0 and (src_tokens != 0).any()

        for batch_idx in range(batch_size):
            if has_prompt:
                output_tokens = src_tokens[batch_idx].tolist()
            else:
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

                # Apply temperature scaling
                logits = logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(-1, top_k_indices, top_k_logits)

                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(
                        probs, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token (the first one)
                    sorted_indices_to_remove[..., 0] = False

                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    # Renormalize
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1).item()
                output_tokens.append(next_token)

                if next_token == eos_idx:
                    break

            results.append(torch.tensor(output_tokens, device=device))
        return results
