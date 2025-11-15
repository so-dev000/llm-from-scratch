import regex


class Tokenizer:
    def __init__(self, pattern=None):
        self.merges = {}  # (int, int) -> int
        self.vocab = {}  # int -> bytes
        gpt2_pattern = (
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?"
            r"[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )
        self.compiled_pattern = regex.compile(pattern or gpt2_pattern)

    def _split_text(self, text):
        return self.compiled_pattern.findall(text)

    def train(self, texts, vocab_size):
        # Pre-tokenize: split by pattern, then encode each chunk
        all_ids = []
        for text in texts:
            chunks = self._split_text(text)
            for chunk in chunks:
                all_ids.append(list(chunk.encode("utf-8")))

        # initialize vocab
        # 1 byte, 8 bits, 0~255
        for i in range(256):
            self.vocab[i] = bytes([i])

        num_merges = vocab_size - 256
        for i in range(num_merges):
            # collect stats from all chunks
            stats = {}
            for ids in all_ids:
                for pair, count in self._get_stats(ids).items():
                    stats[pair] = stats.get(pair, 0) + count
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # Apply merge to each chunk
            all_ids = [self._merge(ids, pair, idx) for ids in all_ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        # Pre-tokenize: split by pattern, then encode each chunk
        chunks = self._split_text(text)
        tokens = []
        for chunk in chunks:
            chunk_tokens = list(chunk.encode("utf-8"))
            # Apply learned merges to each chunk
            while len(chunk_tokens) >= 2:
                stats = self._get_stats(chunk_tokens)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break  # no merge available
                idx = self.merges[pair]
                chunk_tokens = self._merge(chunk_tokens, pair, idx)
            tokens.extend(chunk_tokens)
        return tokens

    def decode(self, ids):
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
