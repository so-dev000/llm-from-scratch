from tokenizer.bpe import BPE


class TestBPE:
    def test_train(self):
        tokenizer = BPE()
        tokenizer.train(["hello world"], vocab_size=300)

        assert len(tokenizer.vocab) >= 256
        assert len(tokenizer.merges) > 0
        assert len(tokenizer.vocab) == 4 + 256 + len(tokenizer.merges)

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
        tokenizer.train(["aaa"], vocab_size=261)

        assert len(tokenizer.merges) == 1
        assert len(tokenizer.vocab) == 261

    def test_special_tokens_encode(self):
        tokenizer = BPE()
        tokenizer.train(["hello"], vocab_size=300)

        encoded = tokenizer.encode("hello", add_special_tokens=True)

        assert encoded[0] == 2  # <BOS>
        assert encoded[-1] == 3  # <EOS>

    def test_special_tokens_decode(self):
        tokenizer = BPE()
        tokenizer.train(["hello"], vocab_size=300)

        encoded = tokenizer.encode("hello", add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        assert decoded == "hello"
        assert "<BOS>" not in decoded
        assert "<EOS>" not in decoded
