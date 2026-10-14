"""Tests for Curator operations: feedback, refine, merge, prune, cap."""

import pytest

from ace_in_a_box import Curator, Playbook, jaccard, tokenize


def seeded(contents, section="strategies"):
    playbook = Playbook()
    curator = Curator(playbook)
    bullets = [curator.add_bullet(section, c) for c in contents]
    return playbook, curator, bullets


# -- tokenize / jaccard -------------------------------------------------------

def test_tokenize_normalises_case_and_punctuation():
    assert tokenize("Run the Tests, please!") == {"run", "the", "tests", "please"}


def test_jaccard_identical_and_disjoint():
    assert jaccard("run tests now", "run tests now") == 1.0
    assert jaccard("run tests", "eat pizza") == 0.0
    assert jaccard("", "") == 1.0
    assert jaccard("", "something") == 0.0


def test_jaccard_partial_overlap():
    # {run, the, tests} vs {run, the, build} -> 2/4
    assert jaccard("run the tests", "run the build") == pytest.approx(0.5)


# -- feedback -------------------------------------------------------------------

def test_apply_feedback_increments_counters():
    _, curator, (bullet,) = seeded(["check logs first"])
    curator.apply_feedback(bullet.id, helpful=True)
    curator.apply_feedback(bullet.id, helpful=True)
    curator.apply_feedback(bullet.id, helpful=False)
    assert bullet.helpful_count == 2
    assert bullet.harmful_count == 1
    assert bullet.net_score == 1


def test_apply_feedback_batch():
    _, curator, bullets = seeded(["a first thing", "b second thing"])
    curator.apply_feedback_batch({bullets[0].id: True, bullets[1].id: False})
    assert bullets[0].helpful_count == 1
    assert bullets[1].harmful_count == 1


def test_apply_feedback_unknown_id_raises():
    _, curator, _ = seeded(["something"])
    with pytest.raises(KeyError):
        curator.apply_feedback("b-9999", helpful=True)


# -- refine -----------------------------------------------------------------------

def test_refine_bullet_updates_text_keeps_counters():
    _, curator, (bullet,) = seeded(["old wording here"])
    curator.apply_feedback(bullet.id, helpful=True)
    curator.refine_bullet(bullet.id, "new   wording\nhere")
    assert bullet.content == "new wording here"
    assert bullet.helpful_count == 1


def test_refine_bullet_rejects_empty():
    _, curator, (bullet,) = seeded(["something"])
    with pytest.raises(ValueError):
        curator.refine_bullet(bullet.id, "   ")


# -- merge --------------------------------------------------------------------------

def test_merge_duplicates_sums_counters_into_most_tested():
    playbook, curator, (a, b, other) = seeded([
        "always run the full test suite before committing",
        "always run the full test suite before committing code",
        "water the plants on tuesdays",
    ])
    curator.apply_feedback(a.id, helpful=True)
    for _ in range(3):
        curator.apply_feedback(b.id, helpful=True)
    curator.apply_feedback(b.id, helpful=False)

    report = curator.merge_duplicates(threshold=0.7)

    assert len(playbook) == 2
    assert len(report) == 1
    survivor = playbook.get(report[0]["kept"])
    assert survivor.id == b.id  # most total feedback wins
    assert survivor.helpful_count == 4  # 1 + 3 summed
    assert survivor.harmful_count == 1
    assert report[0]["absorbed"] == [a.id]
    assert playbook.get(other.id).content == "water the plants on tuesdays"


def test_merge_duplicates_respects_sections():
    playbook, curator, _ = seeded(["same text here"])
    curator.add_bullet("failure_modes", "same text here")
    report = curator.merge_duplicates(threshold=0.5)
    assert report == []
    assert len(playbook) == 2  # identical text, different sections: no merge


def test_merge_duplicates_below_threshold_noop():
    playbook, curator, _ = seeded([
        "deploy on fridays is risky",
        "deploy on mondays is calm",
    ])
    assert curator.merge_duplicates(threshold=0.99) == []
    assert len(playbook) == 2


def test_merge_duplicates_invalid_threshold():
    _, curator, _ = seeded(["x"])
    with pytest.raises(ValueError):
        curator.merge_duplicates(threshold=0)


# -- prune ----------------------------------------------------------------------------

def test_prune_removes_below_threshold_only():
    playbook, curator, (bad, meh, good) = seeded(
        ["bad advice", "meh advice", "good advice"])
    for _ in range(4):
        curator.apply_feedback(bad.id, helpful=False)      # net -4
    curator.apply_feedback(meh.id, helpful=False)          # net -1
    for _ in range(2):
        curator.apply_feedback(good.id, helpful=True)      # net +2

    removed = curator.prune(threshold=-2)

    assert [b.id for b in removed] == [bad.id]
    assert {b.id for b in playbook.bullets} == {meh.id, good.id}


def test_prune_default_keeps_fresh_bullets():
    playbook, curator, _ = seeded(["brand new, untested"])
    assert curator.prune() == []
    assert len(playbook) == 1


# -- cap ---------------------------------------------------------------------------------

def test_enforce_cap_evicts_lowest_value_first():
    playbook, curator, bullets = seeded(
        ["one unique thing", "two unique thing", "three unique thing",
         "four unique thing"])
    curator.apply_feedback(bullets[0].id, helpful=False)   # net -1: worst
    curator.apply_feedback(bullets[1].id, helpful=False)
    curator.apply_feedback(bullets[1].id, helpful=True)    # net 0, 2 votes
    # bullets[2]: net 0, 0 votes -> weaker evidence than bullets[1]
    for _ in range(5):
        curator.apply_feedback(bullets[3].id, helpful=True)  # net +5: best

    evicted = curator.enforce_cap(2)

    assert [b.id for b in evicted] == [bullets[0].id, bullets[2].id]
    assert {b.id for b in playbook.bullets} == {bullets[1].id, bullets[3].id}


def test_enforce_cap_noop_when_within_cap():
    playbook, curator, _ = seeded(["only one"])
    assert curator.enforce_cap(10) == []
    assert len(playbook) == 1


def test_enforce_cap_to_zero():
    playbook, curator, _ = seeded(["a thing", "b thing"])
    evicted = curator.enforce_cap(0)
    assert len(evicted) == 2
    assert len(playbook) == 0
