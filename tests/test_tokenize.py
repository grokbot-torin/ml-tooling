from tensor_toolkit.tokenize import whitespace_tokenize


def test_whitespace_tokenize_string_and_batch():
    assert whitespace_tokenize("hello world  tokens") == ["hello", "world", "tokens"]
    assert whitespace_tokenize("") == []
    assert whitespace_tokenize(["a b", "c"]) == [["a", "b"], ["c"]]




import pytest

from tensor_toolkit.tokenize import Vocabulary


def test_vocabulary_fit_and_unknowns():
    vocab = Vocabulary(min_freq=2)
    vocab.fit(["cat dog cat", "dog bird", ["cat", "mouse"]])
    # cat appears 3 times, dog 2, bird/mouse 1 so they are dropped
    assert vocab.pad_token in vocab.token_to_id
    assert "cat" in vocab.token_to_id and "dog" in vocab.token_to_id
    assert "bird" not in vocab.token_to_id
    ids = vocab.tokens_to_ids(["cat", "bird", "dog"])
    assert ids[0] == vocab.token_to_id["cat"]
    assert ids[1] == vocab.token_to_id["<unk>"]
    assert vocab.ids_to_tokens(ids)[0] == "cat"
    assert len(vocab) == 4  # pad, unk, cat, dog
