class Tokenizer:
    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {}  # int -> bytes

    def train(self, texts, vocab_size):
        # process each text independently
        all_ids = [list(text.encode("utf-8")) for text in texts]

        # initialize vocab
        # 1 byte, 8 bits, 0~255
        for i in range(256):
            self.vocab[i] = bytes([i])

        num_merges = vocab_size - 256
        for i in range(num_merges):
            # collect stats from all texts independently
            stats = {}
            for ids in all_ids:
                for pair, count in self._get_stats(ids).items():
                    stats[pair] = stats.get(pair, 0) + count
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # Apply merge to each text independently
            all_ids = [self._merge(ids, pair, idx) for ids in all_ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no merge available
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
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
