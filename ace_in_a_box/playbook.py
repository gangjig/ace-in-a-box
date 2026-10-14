"""Playbook data model for ace-in-a-box.

A *playbook* is the structured, evolving context an agent carries between
runs (the core idea of Agentic Context Engineering, ACE).  It is a list of
*bullets*: small, single-purpose pieces of guidance grouped into sections.
Every bullet tracks how often it proved helpful or harmful, which is the
feedback signal the Curator uses to keep the playbook healthy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

#: Canonical section names, in display order.  Custom sections are allowed;
#: they render after the canonical ones, alphabetically.
SECTIONS: tuple[str, ...] = (
    "strategies",
    "domain_facts",
    "failure_modes",
    "code_snippets",
)

_ID_SUFFIX_RE = re.compile(r"(\d+)$")


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(text: str) -> str:
    """Collapse all whitespace runs (including newlines) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Bullet:
    """One playbook entry: a piece of guidance plus its feedback counters."""

    id: str
    section: str
    content: str
    helpful_count: int = 0
    harmful_count: int = 0
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)

    def __post_init__(self) -> None:
        self.content = clean_text(self.content)
        if not self.content:
            raise ValueError("bullet content must not be empty")
        if self.helpful_count < 0 or self.harmful_count < 0:
            raise ValueError("feedback counters must not be negative")

    # -- derived signals ----------------------------------------------------
    @property
    def net_score(self) -> int:
        """helpful minus harmful; the primary quality signal."""
        return self.helpful_count - self.harmful_count

    @property
    def total_feedback(self) -> int:
        """Total votes cast on this bullet (evidence strength)."""
        return self.helpful_count + self.harmful_count

    def value_key(self) -> tuple:
        """Sort key for value: higher tuples are *more* valuable bullets.

        Ordered by net score, then amount of evidence, then recency, so a
        well-tested, recently-confirmed bullet outranks a speculative one.
        The trailing id keeps ordering fully deterministic.
        """
        return (self.net_score, self.total_feedback, self.updated_at, self.id)

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "section": self.section,
            "content": self.content,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Bullet":
        return cls(
            id=str(data["id"]),
            section=str(data["section"]),
            content=str(data["content"]),
            helpful_count=int(data.get("helpful_count", 0)),
            harmful_count=int(data.get("harmful_count", 0)),
            created_at=str(data.get("created_at") or utcnow_iso()),
            updated_at=str(data.get("updated_at") or utcnow_iso()),
        )


class Playbook:
    """An insertion-ordered collection of bullets with id allocation."""

    def __init__(self, bullets: list[Bullet] | None = None) -> None:
        self._bullets: dict[str, Bullet] = {}
        self._next_id = 1
        for bullet in bullets or []:
            self.insert(bullet)

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self._bullets)

    def __contains__(self, bullet_id: str) -> bool:
        return bullet_id in self._bullets

    def __iter__(self):
        return iter(self._bullets.values())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Playbook):
            return NotImplemented
        return self.bullets == other.bullets

    def __repr__(self) -> str:
        return f"Playbook({len(self)} bullets)"

    @property
    def bullets(self) -> list[Bullet]:
        """All bullets, in insertion order."""
        return list(self._bullets.values())

    # -- access ---------------------------------------------------------------
    def get(self, bullet_id: str) -> Bullet:
        try:
            return self._bullets[bullet_id]
        except KeyError:
            raise KeyError(f"no bullet with id {bullet_id!r}") from None

    def by_section(self, section: str) -> list[Bullet]:
        return [b for b in self._bullets.values() if b.section == section]

    def sections(self) -> list[str]:
        """Sections present, canonical order first, custom ones after."""
        present = {b.section for b in self._bullets.values()}
        ordered = [s for s in SECTIONS if s in present]
        ordered += sorted(present - set(SECTIONS))
        return ordered

    # -- mutation (used by the Curator and persistence loaders) ---------------
    def allocate_id(self) -> str:
        bullet_id = f"b-{self._next_id:04d}"
        self._next_id += 1
        return bullet_id

    def insert(self, bullet: Bullet) -> Bullet:
        if bullet.id in self._bullets:
            raise ValueError(f"duplicate bullet id {bullet.id!r}")
        self._bullets[bullet.id] = bullet
        match = _ID_SUFFIX_RE.search(bullet.id)
        if match:  # keep future allocations collision-free after a load
            self._next_id = max(self._next_id, int(match.group(1)) + 1)
        return bullet

    def remove(self, bullet_id: str) -> Bullet:
        try:
            return self._bullets.pop(bullet_id)
        except KeyError:
            raise KeyError(f"no bullet with id {bullet_id!r}") from None
