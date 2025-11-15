from component.tokenizer import Tokenizer


class TestTokenizer:
    def test_train(self):
        tokenizer = Tokenizer()
        tokenizer.train(["hello world"], vocab_size=300)

        assert len(tokenizer.vocab) >= 256
        assert len(tokenizer.merges) > 0

    def test_encode_decode_roundtrip(self):
        tokenizer = Tokenizer()
        tokenizer.train(["hello world"], vocab_size=300)

        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)

        assert decoded == "hello"

    def test_unicode(self):
        tokenizer = Tokenizer()
        tokenizer.train(["こんにちは"], vocab_size=300)

        encoded = tokenizer.encode("こんにちは")
        decoded = tokenizer.decode(encoded)

        assert decoded == "こんにちは"

    def test_merge_reduces_tokens(self):
        tokenizer = Tokenizer()
        tokenizer.train(["aaaa" * 100], vocab_size=300)

        encoded = tokenizer.encode("aaaa")

        assert len(encoded) < 4

    def test_encode_unseen_pairs(self):
        tokenizer = Tokenizer()
        tokenizer.train(["aaa"], vocab_size=260)

        encoded = tokenizer.encode("bbb")

        assert len(encoded) == 3
