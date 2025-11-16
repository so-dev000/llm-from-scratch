from tokenizer.bpe import BPE


class TestBPE:
    def test_train(self):
        tokenizer = BPE()
        tokenizer.train(["hello world"], vocab_size=300)

        assert len(tokenizer.vocab) >= 256
        assert len(tokenizer.merges) > 0
        assert len(tokenizer.vocab) == 256 + len(tokenizer.merges)

    def test_encode_decode_roundtrip(self):
        tokenizer = BPE()
        tokenizer.train(["hello world"], vocab_size=300)

        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)

        assert decoded == "hello"

    def test_unicode(self):
        tokenizer = BPE()
        tokenizer.train(["こんにちは"], vocab_size=300)

        encoded = tokenizer.encode("こんにちは")
        decoded = tokenizer.decode(encoded)

        assert decoded == "こんにちは"

    def test_merge_reduces_tokens(self):
        tokenizer = BPE()
        tokenizer.train(["aaaa" * 100], vocab_size=300)

        encoded = tokenizer.encode("aaaa")

        assert len(encoded) < 4
        assert len(encoded) >= 1

    def test_encode_unseen_pairs(self):
        tokenizer = BPE()
        tokenizer.train(["aaa"], vocab_size=260)

        encoded = tokenizer.encode("bbb")

        assert len(encoded) == 3

    def test_pretokenize_splits_text(self):
        tokenizer = BPE()
        chunks = tokenizer._split_text("Hello, world!")

        assert "".join(chunks) == "Hello, world!"

    def test_pretokenize_preserves_boundaries(self):
        tokenizer = BPE()
        tokenizer.train(["cat", "dog"], vocab_size=300)

        encoded = tokenizer.encode("cat dog")
        decoded = tokenizer.decode(encoded)

        assert decoded == "cat dog"

    def test_empty_text(self):
        tokenizer = BPE()
        tokenizer.train(["hello"], vocab_size=300)

        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""

    def test_complete_all_merges(self):
        tokenizer = BPE()
        tokenizer.train(["aaa"], vocab_size=257)

        assert len(tokenizer.merges) == 1
        assert len(tokenizer.vocab) == 257
