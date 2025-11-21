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

    def _split_text(self, text):
        return self.compiled_pattern.findall(text)

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
            chunks = self._split_text(text)
            for chunk in chunks:
                byte_vals = list(chunk.encode("utf-8"))
                chunk_ids = [num_special + b for b in byte_vals]
                all_ids.append(chunk_ids)

        num_merges = vocab_size - 256 - num_special
        for i in tqdm(range(num_merges), desc="Training BPE"):
            # collect stats from all chunks
            stats = {}
            for ids in all_ids:
                for pair, count in self._get_stats(ids).items():
                    stats[pair] = stats.get(pair, 0) + count
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = num_special + 256 + i
            # Apply merge to each chunk
            all_ids = [self._merge(ids, pair, idx) for ids in all_ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text, add_special_tokens=False):
        num_special = len(self.special_tokens)

        # Pre-tokenize: split by pattern, then encode each chunk
        chunks = self._split_text(text)
        tokens = []

        if add_special_tokens:
            tokens.append(self.special_tokens["<BOS>"])

        for chunk in chunks:
            byte_vals = list(chunk.encode("utf-8"))
            chunk_tokens = [num_special + b for b in byte_vals]

            # Apply learned merges to each chunk
            while len(chunk_tokens) >= 2:
                stats = self._get_stats(chunk_tokens)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break  # no merge available
                idx = self.merges[pair]
                chunk_tokens = self._merge(chunk_tokens, pair, idx)
            tokens.extend(chunk_tokens)

        if add_special_tokens:
            tokens.append(self.special_tokens["<EOS>"])

        return tokens

    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            special_ids = set(self.special_tokens.values())
            ids = [i for i in ids if i not in special_ids]

        tokens = b"".join(self.vocab[i] for i in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _get_stats(self, ids):
        # get adjacent pair counts
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

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
