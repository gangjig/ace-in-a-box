"""Render a playbook into a prompt-ready string, under a size budget.

The rendered playbook is what actually goes into the agent's context, so it
must respect a budget.  When the playbook does not fit, the *lowest-value*
bullets (per :meth:`Bullet.value_key`) are dropped first — the playbook
degrades gracefully instead of being truncated mid-sentence.
"""

from __future__ import annotations

from .playbook import Playbook

#: Rough characters-per-token ratio used when a token budget is given.
CHARS_PER_TOKEN = 4


def _bullet_line(bullet) -> str:
    return (
        f"- [{bullet.id}] (+{bullet.helpful_count}/-{bullet.harmful_count}) "
        f"{bullet.content}"
    )


def _assemble(playbook: Playbook, included_ids: set[str]) -> str:
    lines = ["# Playbook", ""]
    for section in playbook.sections():
        bullets = [b for b in playbook.by_section(section) if b.id in included_ids]
        if not bullets:
            continue
        bullets.sort(key=lambda b: b.value_key(), reverse=True)
        lines.append(f"## {section}")
        lines.append("")
        lines.extend(_bullet_line(b) for b in bullets)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_prompt(
    playbook: Playbook,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
) -> str:
    """Render ``playbook`` for inclusion in a prompt.

    ``max_tokens`` is converted to characters at ``CHARS_PER_TOKEN``
    chars/token (a conservative approximation for English text); if both
    budgets are given, the tighter one wins.  With no budget, every bullet
    is rendered.  Bullets are *selected* by value (highest first) but
    *displayed* grouped by section.
    """
    budget = max_chars
    if max_tokens is not None:
        token_budget = max_tokens * CHARS_PER_TOKEN
        budget = token_budget if budget is None else min(budget, token_budget)

    all_ids = {b.id for b in playbook.bullets}
    if budget is None:
        return _assemble(playbook, all_ids)

    # Greedily keep highest-value bullets while the render fits the budget.
    ranked = sorted(playbook.bullets, key=lambda b: b.value_key(), reverse=True)
    kept: set[str] = set()
    for bullet in ranked:
        trial = kept | {bullet.id}
        if len(_assemble(playbook, trial)) <= budget:
            kept = trial
    dropped = len(all_ids) - len(kept)
    if dropped:
        # Make room for the omission note itself: sacrifice the lowest-value
        # kept bullets until text + note fits the budget.
        def note(n: int) -> str:
            return f"<!-- {n} lower-value bullet(s) omitted to fit the budget -->\n"

        while kept and len(_assemble(playbook, kept)) + len(note(dropped)) > budget:
            victim = min(
                (b for b in playbook.bullets if b.id in kept),
                key=lambda b: b.value_key(),
            )
            kept.discard(victim.id)
            dropped += 1
        text = _assemble(playbook, kept)
        if len(text) + len(note(dropped)) <= budget:
            text += note(dropped)
        return text
    return _assemble(playbook, kept)
