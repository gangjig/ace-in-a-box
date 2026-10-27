"""Tests for the monitor-holdout split and the update guard."""

import random

import pytest

from ace_in_a_box.evalset import (
    HoldoutGuard,
    HoldoutLeakError,
    split_cases,
)


def _cases(n=40):
    return [{"id": f"case-{i:02d}", "task": f"task {i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def test_split_is_deterministic():
    first = split_cases(_cases())
    second = split_cases(_cases())
    assert first.drive_ids == second.drive_ids
    assert first.holdout_ids == second.holdout_ids
    assert first.fingerprint == second.fingerprint


def test_split_is_order_independent():
    cases = _cases()
    shuffled = cases[:]
    random.Random(1234).shuffle(shuffled)
    split = split_cases(cases)
    reshuffled = split_cases(shuffled)
    assert split.drive_ids == reshuffled.drive_ids
    assert split.holdout_ids == reshuffled.holdout_ids
    assert split.fingerprint == reshuffled.fingerprint


def test_case_never_migrates_when_eval_set_grows():
    subset = split_cases(_cases(20))
    superset = split_cases(_cases(40))
    for case_id in subset.drive_ids:
        assert case_id in superset.drive_ids
    for case_id in subset.holdout_ids:
        assert case_id in superset.holdout_ids


def test_fraction_moves_cases_one_way_only():
    small = split_cases(_cases(), holdout_fraction=0.3)
    mid = split_cases(_cases(), holdout_fraction=0.5)
    large = split_cases(_cases(), holdout_fraction=0.7)
    assert small.holdout_ids <= mid.holdout_ids <= large.holdout_ids
    assert len(small.holdout) < len(large.holdout)


def test_fingerprint_tracks_the_ruler():
    base = split_cases(_cases(20))
    assert base.fingerprint != split_cases(_cases(21)).fingerprint
    assert base.fingerprint != split_cases(_cases(20), seed="other").fingerprint
    assert base.fingerprint != split_cases(
        _cases(20), holdout_fraction=0.25
    ).fingerprint


def test_split_partitions_every_case_exactly_once():
    cases = _cases()
    split = split_cases(cases)
    assert split.drive_ids | split.holdout_ids == {c["id"] for c in cases}
    assert not (split.drive_ids & split.holdout_ids)
    assert len(split.drive) + len(split.holdout) == len(cases)
    # Non-id fields pass through untouched for the caller's runner.
    assert all("task" in c for c in split.drive + split.holdout)


def test_split_rejects_empty_case_list():
    with pytest.raises(ValueError, match="empty"):
        split_cases([])


def test_split_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        split_cases([{"id": "a"}, {"id": "b"}, {"id": "a"}])


def test_split_rejects_missing_id():
    with pytest.raises(ValueError, match="'id'"):
        split_cases([{"id": "a"}, {"task": "no id here"}])


def test_split_rejects_out_of_range_fraction():
    with pytest.raises(ValueError, match="holdout_fraction"):
        split_cases(_cases(), holdout_fraction=1.5)


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def test_guard_blocks_holdout_case():
    split = split_cases(_cases())
    guard = HoldoutGuard(split)
    holdout_id = next(iter(split.holdout_ids))
    with pytest.raises(HoldoutLeakError, match="holdout"):
        guard.check(holdout_id)


def test_guard_allows_drive_case():
    split = split_cases(_cases())
    guard = HoldoutGuard(split)
    drive_id = next(iter(split.drive_ids))
    assert guard.check(drive_id) is True
    assert not guard.is_holdout(drive_id)


def test_guard_allows_case_outside_the_eval_set():
    split = split_cases(_cases())
    guard = HoldoutGuard(split)
    # An ordinary production trace is not the eval set's business.
    assert guard.check("prod-trace-98765") is True
