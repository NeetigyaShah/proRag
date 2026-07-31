"""Pure-function tests for the Phase 6 sweep harness: combination generation
only — running a sweep needs a live DB + LLM (§8 Phase 6)."""

from scripts.sweep import generate_combinations


def test_generate_combinations_cartesian_product():
    grid = {"a": [1, 2], "b": [10, 20]}
    combos = generate_combinations(grid)
    assert len(combos) == 4
    assert {"a": 1, "b": 10} in combos
    assert {"a": 2, "b": 20} in combos


def test_generate_combinations_single_key():
    combos = generate_combinations({"x": [1, 2, 3]})
    assert combos == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_generate_combinations_empty_grid():
    assert generate_combinations({}) == [{}]


def test_generate_combinations_covers_full_grid_size():
    grid = {"a": [1, 2], "b": [10, 20, 30], "c": [True, False]}
    combos = generate_combinations(grid)
    assert len(combos) == 2 * 3 * 2
    assert len({tuple(sorted(c.items())) for c in combos}) == len(combos)  # all unique
