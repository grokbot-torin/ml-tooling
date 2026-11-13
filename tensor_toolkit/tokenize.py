"""Lightweight tokenization helpers."""

from __future__ import annotations

from collections.abc import Iterable


def whitespace_tokenize(text: str | Iterable[str]) -> list[str] | list[list[str]]:
    """Split on whitespace, or map that split over a batch of strings."""
    if isinstance(text, str):
        return text.split()
    return [str(item).split() for item in text]

