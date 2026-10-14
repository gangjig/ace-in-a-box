"""Tests for the Bullet / Playbook data model."""

import pytest

from ace_in_a_box import Bullet, Playbook
from ace_in_a_box.playbook import SECTIONS


def make_bullet(bid="b-0001", section="strategies", content="do the thing",
                helpful=0, harmful=0):
    return Bullet(id=bid, section=section, content=content,
                  helpful_count=helpful, harmful_count=harmful)


def test_bullet_net_score_and_total_feedback():
    bullet = make_bullet(helpful=5, harmful=2)
    assert bullet.net_score == 3
    assert bullet.total_feedback == 7


def test_bullet_content_is_whitespace_normalised():
    bullet = make_bullet(content="  run the\n\ttests   first ")
    assert bullet.content == "run the tests first"


def test_bullet_empty_content_rejected():
    with pytest.raises(ValueError):
        make_bullet(content="   \n ")


def test_bullet_negative_counters_rejected():
    with pytest.raises(ValueError):
        make_bullet(helpful=-1)


def test_playbook_allocates_sequential_ids():
    playbook = Playbook()
    from ace_in_a_box import Curator
    curator = Curator(playbook)
    first = curator.add_bullet("strategies", "one")
    second = curator.add_bullet("strategies", "two")
    assert (first.id, second.id) == ("b-0001", "b-0002")
    assert len(playbook) == 2


def test_playbook_get_missing_raises_key_error():
    with pytest.raises(KeyError):
        Playbook().get("b-9999")


def test_playbook_insert_duplicate_id_rejected():
    playbook = Playbook([make_bullet()])
    with pytest.raises(ValueError):
        playbook.insert(make_bullet())


def test_playbook_id_allocation_continues_after_load():
    playbook = Playbook([make_bullet(bid="b-0007")])
    assert playbook.allocate_id() == "b-0008"


def test_playbook_sections_canonical_order_then_custom():
    playbook = Playbook([
        make_bullet("b-0001", section="zebra"),
        make_bullet("b-0002", section="failure_modes"),
        make_bullet("b-0003", section="strategies"),
    ])
    assert playbook.sections() == ["strategies", "failure_modes", "zebra"]
    assert set(SECTIONS) >= {"strategies", "domain_facts",
                             "failure_modes", "code_snippets"}


def test_playbook_equality():
    a = Playbook([make_bullet()])
    b = Playbook([make_bullet()])
    # timestamps differ by construction path, so compare structure instead
    assert [x.to_dict()["content"] for x in a.bullets] == \
           [x.to_dict()["content"] for x in b.bullets]
