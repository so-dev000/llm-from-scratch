import pickle
from collections import Counter

import regex
from tqdm import tqdm


class BPE:
    GPT2_PATTERN = (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?"
        r"[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )
    DEFAULT_SPECIAL_TOKENS = {
        "<PAD>": 0,  # padding
        "<UNK>": 1,  # unknown
        "<BOS>": 2,  # beginning of sentence
        "<EOS>": 3,  # end of sentence
    }

    def __init__(self, pattern=None, special_tokens=None):
        self.merges = {}  # (int, int) -> int
        self.vocab = {}  # int -> bytes
        # initialize special tokens
        self.special_tokens = special_tokens or self.DEFAULT_SPECIAL_TOKENS.copy()
        self.compiled_pattern = regex.compile(pattern or self.GPT2_PATTERN)

    def train(self, texts, vocab_size):
        num_special = len(self.special_tokens)
        # initialize vocab
        # special token
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

        # byte token
        # 1 byte, 8 bits, 0~255
        for i in range(256):
            self.vocab[num_special + i] = bytes([i])

        # Pre-tokenize: split by pattern, then encode each chunk
        all_ids = []
        for text in texts:
            chunks = self.compiled_pattern.findall(text)
            for chunk in chunks:
                byte_vals = list(chunk.encode("utf-8"))
                chunk_ids = [num_special + b for b in byte_vals]
                all_ids.append(chunk_ids)

        # count pairs
        pair_counts = Counter()
        for chunk in all_ids:
            pair_counts.update(zip(chunk, chunk[1:]))

        num_merges = vocab_size - 256 - num_special
        for i in tqdm(range(num_merges), desc="Training BPE"):
            if not pair_counts:
                break
            pair = max(pair_counts, key=pair_counts.get)
            new_idx = num_special + 256 + i

            self.merges[pair] = new_idx
            self.vocab[new_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # Apply merge and update pair counts
            self._apply_merge_and_update_pair_counts(
                all_ids, pair, new_idx, pair_counts
            )

    def _apply_merge_and_update_pair_counts(self, chunks, pair, new_idx, pair_counts):
        """
        Example: merge (A, B) → X

        ...P A B Q...
              ↓
         ...P X Q...

        reduce: (P, A) (A, B) (B, Q)
        increase: (P, X) (X, Q)
        """
        p0, p1 = pair
        # delete merged pair
        del pair_counts[pair]

        for chunk in chunks:
            i = 0
            while i < len(chunk) - 1:
                if chunk[i] == p0 and chunk[i + 1] == p1:
                    if i > 0:
                        # reduce left pair (P, A)
                        pair_counts[(chunk[i - 1], p0)] -= 1
                    if i + 2 < len(chunk):
                        # reduce right pair (B, Q)
                        pair_counts[(p1, chunk[i + 2])] -= 1
                    # merge
                    chunk[i : i + 2] = [new_idx]

                    if i > 0:
                        # increase new pair (P, X)
                        pair_counts[(chunk[i - 1], new_idx)] += 1
                    if i + 1 < len(chunk):
                        # increase new pair (X, Q)
                        pair_counts[(new_idx, chunk[i + 1])] += 1
                else:
                    i += 1
        # delete non-positive count pairs
        pair_counts += Counter()

    def encode(self, text, add_special_tokens=False):
        num_special = len(self.special_tokens)

        # Pre-tokenize: split by pattern, then encode each chunk
        chunks = self.compiled_pattern.findall(text)
        tokens = []

        if add_special_tokens:
            tokens.append(self.special_tokens["<BOS>"])

        for chunk in chunks:
            byte_vals = list(chunk.encode("utf-8"))
            chunk_tokens = [num_special + b for b in byte_vals]

            # Apply learned merges to each chunk
            while len(chunk_tokens) >= 2:
                stats = set(zip(chunk_tokens, chunk_tokens[1:]))
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break  # no merge available
                idx = self.merges[pair]
                chunk_tokens = self._merge(chunk_tokens, pair, idx)
            tokens.extend(chunk_tokens)

        if add_special_tokens:
            tokens.append(self.special_tokens["<EOS>"])

        return tokens

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            special_ids = set(self.special_tokens.values())
            ids = [i for i in ids if i not in special_ids]

        tokens = b"".join(self.vocab[i] for i in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "merges": self.merges,
                    "vocab": self.vocab,
                    "special_tokens": self.special_tokens,
                    "pattern": self.compiled_pattern.pattern,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        tokenizer = cls(pattern=data["pattern"], special_tokens=data["special_tokens"])
        tokenizer.merges = data["merges"]
        tokenizer.vocab = data["vocab"]
        return tokenizer

    def get_vocab_size(self):
        return len(self.vocab)
