"""Unit tests for the FSDD data layer and the CTC vocabulary."""
import torch

from src.data import (
    BLANK_ID,
    DIGIT_TOKENS,
    VOCAB_SIZE,
    greedy_ctc_decode,
    ids_to_str,
    label_to_id,
)


def test_vocab_size_is_blank_plus_ten_digits():
    assert VOCAB_SIZE == 11
    assert BLANK_ID == 0
    assert DIGIT_TOKENS == [str(d) for d in range(10)]


def test_label_to_id_offsets_by_one():
    for d in range(10):
        assert label_to_id(str(d)) == d + 1


def test_ids_to_str_skips_blank():
    assert ids_to_str([1, 2, 3]) == "012"
    assert ids_to_str([BLANK_ID, 1, BLANK_ID, 2]) == "01"
    assert ids_to_str([BLANK_ID, BLANK_ID]) == ""
    assert ids_to_str([]) == ""


def _one_hot_logits(seq: list[int]) -> torch.Tensor:
    """Build (T, V) logits where each step strongly favors the given id."""
    logits = torch.full((len(seq), VOCAB_SIZE), -10.0)
    for t, idx in enumerate(seq):
        logits[t, idx] = 10.0
    return logits


def test_greedy_decode_collapses_duplicates():
    # blank-separated repeats stay; consecutive duplicates collapse.
    logits = _one_hot_logits([1, 1, BLANK_ID, 1, 2, 2, BLANK_ID, 3])
    assert greedy_ctc_decode(logits) == [1, 1, 2, 3]


def test_greedy_decode_drops_blanks_only():
    logits = _one_hot_logits([BLANK_ID, BLANK_ID, BLANK_ID])
    assert greedy_ctc_decode(logits) == []


def test_greedy_decode_handles_no_blanks_or_duplicates():
    logits = _one_hot_logits([1, 2, 3, 4, 5])
    assert greedy_ctc_decode(logits) == [1, 2, 3, 4, 5]
