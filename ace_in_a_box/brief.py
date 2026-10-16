"""A per-task brief prepended to a rendered playbook prompt.

The brief format follows the prompting formula shared by Anthropic's
Boris Cherny — say what to do, how to verify it, and how much effort to
spend — as reported in Zvi Mowshowitz's AI #190.  A brief is per-task
and render-time only: it is never stored in the playbook file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskBrief:
    """What this task is, how to check it, and how hard to work on it.

    ``goal`` is required and must be non-blank; ``verify`` (how to tell
    the work is done right) and ``effort`` (how much effort to spend)
    are optional and simply omitted from the render when not given.
    """

    goal: str
    verify: str | None = None
    effort: str | None = None

    def __post_init__(self) -> None:
        if not self.goal or not self.goal.strip():
            raise ValueError("TaskBrief.goal must be a non-empty string")


def render_brief(brief: TaskBrief) -> str:
    """Render ``brief`` as a Markdown header block.

    The block ends with a trailing blank line so it composes directly in
    front of the ``# Playbook`` body produced by :mod:`.render`.
    """
    lines = ["# Task brief", "", f"**Goal:** {brief.goal}"]
    if brief.verify is not None:
        lines.append(f"**Verify:** {brief.verify}")
    if brief.effort is not None:
        lines.append(f"**Effort:** {brief.effort}")
    lines.append("")
    return "\n".join(lines) + "\n"
