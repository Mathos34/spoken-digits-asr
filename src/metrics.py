"""Character error rate (Levenshtein distance / target length)."""
from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def cer(refs: list[str], hyps: list[str]) -> float:
    total_dist = sum(levenshtein(r, h) for r, h in zip(refs, hyps, strict=False))
    total_chars = sum(max(1, len(r)) for r in refs)
    return total_dist / total_chars
