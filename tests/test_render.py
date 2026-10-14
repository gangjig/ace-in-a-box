"""Tests for budget-aware prompt rendering."""

from ace_in_a_box import Curator, Playbook, render_prompt


def sample_playbook() -> Playbook:
    playbook = Playbook()
    curator = Curator(playbook)
    good = curator.add_bullet(
        "strategies", "Always check the exit code of shell commands")
    bad = curator.add_bullet(
        "strategies", "Ignore compiler warnings, they are usually fine")
    curator.add_bullet("domain_facts", "Prod deploys happen from the main branch")
    for _ in range(5):
        curator.apply_feedback(good.id, helpful=True)
    for _ in range(5):
        curator.apply_feedback(bad.id, helpful=False)
    return playbook


def test_render_without_budget_includes_everything():
    text = render_prompt(sample_playbook())
    assert "Always check the exit code" in text
    assert "Ignore compiler warnings" in text
    assert "Prod deploys happen" in text
    assert text.startswith("# Playbook")


def test_render_groups_by_section():
    text = render_prompt(sample_playbook())
    assert text.index("## strategies") < text.index("## domain_facts")


def test_render_over_budget_drops_lowest_value_first():
    playbook = sample_playbook()
    full = render_prompt(playbook)
    budget = len(full) - 40  # room for two bullets, not all three
    text = render_prompt(playbook, max_chars=budget)
    assert len(text) <= budget
    assert "Always check the exit code" in text       # highest value: kept
    assert "Ignore compiler warnings" not in text     # lowest value: dropped
    assert "omitted to fit the budget" in text


def test_render_tiny_budget_keeps_header_only():
    text = render_prompt(sample_playbook(), max_chars=30)
    assert len(text) <= 30
    assert text.startswith("# Playbook")


def test_render_max_tokens_uses_char_conversion():
    playbook = sample_playbook()
    via_tokens = render_prompt(playbook, max_tokens=10)  # -> 40 chars
    assert len(via_tokens) <= 40
    via_chars = render_prompt(playbook, max_chars=40)
    assert via_tokens == via_chars


def test_render_empty_playbook():
    assert render_prompt(Playbook()) == "# Playbook\n"
