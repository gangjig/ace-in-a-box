"""Tests for the task-brief header in prompt rendering."""

import dataclasses

import pytest

from ace_in_a_box import Curator, Playbook, TaskBrief, render_brief, render_prompt


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


def bullet_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("- [")]


# -- render_brief -----------------------------------------------------------

def test_render_brief_all_fields():
    brief = TaskBrief(
        goal="Fix the parser",
        verify="the full test suite passes",
        effort="a focused hour",
    )
    assert render_brief(brief) == (
        "# Task brief\n"
        "\n"
        "**Goal:** Fix the parser\n"
        "**Verify:** the full test suite passes\n"
        "**Effort:** a focused hour\n"
        "\n"
    )


def test_render_brief_goal_only():
    assert render_brief(TaskBrief(goal="Fix the parser")) == (
        "# Task brief\n\n**Goal:** Fix the parser\n\n"
    )


def test_render_brief_verify_without_effort():
    text = render_brief(TaskBrief(goal="Fix it", verify="tests pass"))
    assert "**Verify:** tests pass" in text
    assert "**Effort:**" not in text


def test_render_brief_effort_without_verify():
    text = render_brief(TaskBrief(goal="Fix it", effort="one hour"))
    assert "**Effort:** one hour" in text
    assert "**Verify:**" not in text


@pytest.mark.parametrize("goal", ["", "   ", "\n\t"])
def test_brief_blank_goal_raises(goal):
    with pytest.raises(ValueError):
        TaskBrief(goal=goal)


def test_brief_is_frozen():
    brief = TaskBrief(goal="Fix it")
    with pytest.raises(dataclasses.FrozenInstanceError):
        brief.goal = "Something else"


# -- render_prompt with a brief ----------------------------------------------

def test_render_with_brief_puts_brief_before_playbook():
    brief = TaskBrief(goal="Ship the fix", verify="tests pass")
    text = render_prompt(sample_playbook(), brief=brief)
    assert text.startswith("# Task brief")
    assert text.index("# Task brief") < text.index("# Playbook")
    assert "**Goal:** Ship the fix" in text
    assert "Always check the exit code" in text


def test_render_without_brief_is_unchanged():
    playbook = sample_playbook()
    assert render_prompt(playbook, brief=None) == render_prompt(playbook)
    assert not render_prompt(playbook).startswith("# Task brief")


def test_brief_counts_toward_budget():
    playbook = sample_playbook()
    brief = TaskBrief(goal="Ship the fix")
    budget = len(render_prompt(playbook)) - 5  # everything *just* fits
    without = render_prompt(playbook, max_chars=budget)
    with_brief = render_prompt(playbook, brief=brief, max_chars=budget)
    assert len(bullet_lines(without)) == 2
    assert len(bullet_lines(with_brief)) < len(bullet_lines(without))
    assert len(with_brief) <= budget
    assert with_brief.startswith(render_brief(brief))


def test_tight_budget_drops_bullets_but_brief_stays_complete():
    playbook = sample_playbook()
    brief = TaskBrief(
        goal="Ship the fix", verify="tests pass", effort="one hour")
    prefix = render_brief(brief)
    budget = len(prefix) + 30  # room for the playbook header, no bullets
    text = render_prompt(playbook, brief=brief, max_chars=budget)
    assert text.startswith(prefix)          # brief complete, never truncated
    assert "# Playbook" in text
    assert bullet_lines(text) == []         # every bullet dropped instead
    assert len(text) <= budget


def test_brief_alone_exceeding_budget_still_renders_in_full():
    brief = TaskBrief(goal="A goal much longer than the tiny budget allows")
    text = render_prompt(sample_playbook(), brief=brief, max_chars=10)
    assert text.startswith(render_brief(brief))
    assert bullet_lines(text) == []
