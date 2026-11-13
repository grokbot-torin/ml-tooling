from tensor_toolkit.tokenize import whitespace_tokenize


def test_whitespace_tokenize_string_and_batch():
    assert whitespace_tokenize("hello world  tokens") == ["hello", "world", "tokens"]
    assert whitespace_tokenize("") == []
    assert whitespace_tokenize(["a b", "c"]) == [["a", "b"], ["c"]]

