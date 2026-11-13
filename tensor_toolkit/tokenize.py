"""Lightweight tokenization helpers."""

from __future__ import annotations

from collections.abc import Iterable


def whitespace_tokenize(text: str | Iterable[str]) -> list[str] | list[list[str]]:
    """Split on whitespace, or map that split over a batch of strings."""
    if isinstance(text, str):
        return text.split()
    return [str(item).split() for item in text]




class Vocabulary:
    """Build a frequency-filtered token vocabulary with pad and unk ids."""

    def __init__(self, min_freq: int = 1, unk_token: str = "<unk>", pad_token: str = "<pad>") -> None:
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1")
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.token_to_id: dict[str, int] | None = None
        self.id_to_token: list[str] | None = None

    def fit(self, corpus: Iterable) -> "Vocabulary":
        counts: dict[str, int] = {}
        for doc in corpus:
            tokens = whitespace_tokenize(doc) if isinstance(doc, str) else list(doc)
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
        tokens = [self.pad_token, self.unk_token]
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if count >= self.min_freq and token not in tokens:
                tokens.append(token)
        self.id_to_token = tokens
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return self

    def tokens_to_ids(self, tokens: Iterable[str]) -> list[int]:
        if self.token_to_id is None:
            raise RuntimeError("Vocabulary must be fit before tokens_to_ids")
        unk = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(token, unk) for token in tokens]

    def ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        if self.id_to_token is None:
            raise RuntimeError("Vocabulary must be fit before ids_to_tokens")
        return [self.id_to_token[idx] for idx in ids]

    def __len__(self) -> int:
        if self.id_to_token is None:
            return 0
        return len(self.id_to_token)



def ngrams(tokens: Iterable[str], n: int) -> list[tuple[str, ...]]:
    """Return ordered n-grams from a token sequence."""
    if n < 1:
        raise ValueError("n must be >= 1")
    tokens = list(tokens)
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]



def pad_sequences(
    sequences,
    length: int | None = None,
    pad_id: int = 0,
    truncating: str = "post",
    padding: str = "post",
):
    """Pad or truncate a batch of id sequences to a rectangular array."""
    import numpy as np

    seqs = [list(seq) for seq in sequences]
    if length is None:
        length = max((len(seq) for seq in seqs), default=0)
    if length < 0:
        raise ValueError("length must be non-negative")
    if truncating not in {"pre", "post"} or padding not in {"pre", "post"}:
        raise ValueError("truncating and padding must be 'pre' or 'post'")
    out = np.full((len(seqs), length), pad_id, dtype=int)
    for i, seq in enumerate(seqs):
        trimmed = seq[:length] if truncating == "post" else seq[-length:]
        if padding == "post":
            out[i, : len(trimmed)] = trimmed
        else:
            out[i, length - len(trimmed) :] = trimmed
    return out
