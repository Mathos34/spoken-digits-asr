"""Unit tests for Levenshtein distance and Character Error Rate."""
from src.metrics import cer, levenshtein


def test_levenshtein_identical_strings_is_zero():
    assert levenshtein("hello", "hello") == 0
    assert levenshtein("", "") == 0


def test_levenshtein_empty_against_nonempty():
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3


def test_levenshtein_single_substitution():
    assert levenshtein("kitten", "sitten") == 1


def test_levenshtein_classic_example():
    # The Wikipedia canonical example: "kitten" to "sitting" is 3 edits.
    assert levenshtein("kitten", "sitting") == 3


def test_cer_perfect_predictions_is_zero():
    refs = ["123", "4", "789"]
    hyps = ["123", "4", "789"]
    assert cer(refs, hyps) == 0.0


def test_cer_one_substitution_in_one_word():
    # One substitution out of 7 total characters in refs.
    refs = ["1234", "567"]
    hyps = ["1235", "567"]
    assert cer(refs, hyps) == 1 / 7


def test_cer_empty_hypothesis_counts_every_char_as_a_deletion():
    refs = ["12345"]
    hyps = [""]
    assert cer(refs, hyps) == 1.0


def test_cer_handles_empty_reference_via_max_one_denominator():
    # An empty reference contributes 1 to the denominator (max(1, 0)),
    # so the metric is well-defined and stays in a sane range.
    refs = ["", "12"]
    hyps = ["1", "12"]
    # dist = lev("", "1") + lev("12","12") = 1 + 0 = 1; chars = max(1,0) + 2 = 3.
    assert cer(refs, hyps) == 1 / 3
