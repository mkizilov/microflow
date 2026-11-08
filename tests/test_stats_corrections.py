import math

import pytest

from microflow.stats.corrections import adjust, benjamini_hochberg, holm


def test_benjamini_hochberg_matches_a_worked_example() -> None:
    # Sorted: 0.005, 0.01, 0.03, 0.04 with ranks 1..4. Raw p * n / rank: 0.02, 0.02, 0.04, 0.04.
    # Monotone from the largest down keeps them as they are.
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_holm_matches_a_worked_example() -> None:
    # Sorted: 0.005, 0.01, 0.03, 0.04 with factors 4, 3, 2, 1: 0.02, 0.03, 0.06, 0.04.
    # The running maximum gives 0.02, 0.03, 0.06, 0.06. Mapped back to the input order (0.01, 0.04,
    # 0.03, 0.005) that is 0.03, 0.06, 0.06, 0.02.
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_bh_is_never_more_conservative_than_holm() -> None:
    p_values = [0.001, 0.008, 0.039, 0.041, 0.2, 0.6]
    bh = benjamini_hochberg(p_values)
    family_wise = holm(p_values)
    assert all(b <= h + 1e-12 for b, h in zip(bh, family_wise, strict=True))


def test_adjusted_values_are_capped_at_one_and_never_below_the_raw_value() -> None:
    p_values = [0.4, 0.9, 0.95]
    for method in ("bh", "holm"):
        adjusted = adjust(p_values, method)
        assert all(value <= 1.0 for value in adjusted)
        assert all(a >= p - 1e-12 for a, p in zip(adjusted, p_values, strict=True))


def test_missing_p_values_are_excluded_from_the_family() -> None:
    with_gap = adjust([0.01, math.nan, 0.04], "bh")
    without = adjust([0.01, 0.04], "bh")
    assert math.isnan(with_gap[1])
    assert with_gap[0] == pytest.approx(without[0])
    assert with_gap[2] == pytest.approx(without[1])


def test_no_correction_returns_the_raw_values() -> None:
    assert adjust([0.01, 0.2], "none") == [0.01, 0.2]


def test_unknown_correction_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown correction"):
        adjust([0.1], "bonferroni")


def test_an_empty_family_is_returned_empty() -> None:
    assert adjust([], "bh") == []
    assert adjust([], "holm") == []


def test_a_family_of_only_missing_values_stays_missing() -> None:
    adjusted = adjust([math.nan, math.nan], "bh")
    assert all(math.isnan(value) for value in adjusted)
