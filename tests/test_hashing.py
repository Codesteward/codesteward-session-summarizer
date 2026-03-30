from summarizer.hashing import compute_hash


class TestComputeHash:
    def test_deterministic(self):
        assert compute_hash("hello") == compute_hash("hello")

    def test_returns_16_hex_chars(self):
        result = compute_hash("test input")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_inputs_different_hashes(self):
        assert compute_hash("input A") != compute_hash("input B")

    def test_empty_string(self):
        result = compute_hash("")
        assert len(result) == 16

    def test_unicode(self):
        result = compute_hash("日本語テスト")
        assert len(result) == 16
