"""Tests for JSON / Markdown persistence round-trips and file helpers."""

import pytest

from ace_in_a_box import (
    Curator,
    Playbook,
    dumps_json,
    from_markdown,
    load_playbook,
    loads_json,
    save_playbook,
    to_markdown,
)


def sample_playbook() -> Playbook:
    playbook = Playbook()
    curator = Curator(playbook)
    a = curator.add_bullet("strategies", "Run tests before committing")
    b = curator.add_bullet("failure_modes", "Never deploy on a Friday afternoon")
    curator.add_bullet("domain_facts", "The staging DB resets every Sunday")
    curator.apply_feedback(a.id, helpful=True)
    curator.apply_feedback(a.id, helpful=True)
    curator.apply_feedback(b.id, helpful=False)
    return playbook


def test_json_round_trip_is_lossless():
    playbook = sample_playbook()
    restored = loads_json(dumps_json(playbook))
    assert restored.bullets == playbook.bullets  # full dataclass equality
    assert len(restored) == 3


def test_json_round_trip_preserves_id_counter():
    playbook = sample_playbook()
    restored = loads_json(dumps_json(playbook))
    new = Curator(restored).add_bullet("strategies", "another one")
    assert new.id == "b-0004"


def test_markdown_round_trip_preserves_core_fields():
    playbook = sample_playbook()
    restored = from_markdown(to_markdown(playbook))
    key = lambda bs: sorted(
        (b.id, b.section, b.content, b.helpful_count, b.harmful_count)
        for b in bs
    )
    assert key(restored.bullets) == key(playbook.bullets)


def test_markdown_is_human_readable():
    md = to_markdown(sample_playbook())
    assert md.startswith("# Playbook")
    assert "## strategies" in md
    assert "- [b-0001] (+2/-0) Run tests before committing" in md
    assert "- [b-0002] (+0/-1) Never deploy on a Friday afternoon" in md


def test_markdown_without_counters_parses_with_zero_votes():
    text = "# Playbook\n\n## strategies\n\n- [b-0001] plain bullet, no counts\n"
    playbook = from_markdown(text)
    bullet = playbook.get("b-0001")
    assert bullet.helpful_count == 0
    assert bullet.harmful_count == 0


def test_save_and_load_by_suffix(tmp_path):
    playbook = sample_playbook()
    for name in ("pb.json", "pb.md"):
        path = save_playbook(playbook, tmp_path / name)
        restored = load_playbook(path)
        # JSON preserves insertion order; Markdown groups by section, so
        # compare content as a set there and in order for JSON.
        if name.endswith(".json"):
            assert [b.content for b in restored.bullets] == \
                   [b.content for b in playbook.bullets]
        else:
            assert sorted(b.content for b in restored.bullets) == \
                   sorted(b.content for b in playbook.bullets)


def test_unknown_suffix_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_playbook(sample_playbook(), tmp_path / "pb.yaml")
    with pytest.raises(ValueError):
        load_playbook(tmp_path / "pb.yaml")


def test_json_bad_version_rejected():
    with pytest.raises(ValueError):
        loads_json('{"version": 99, "bullets": []}')
