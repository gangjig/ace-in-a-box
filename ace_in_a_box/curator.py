"""The Curator: deterministic operations that keep a playbook healthy.

In ACE terms the Curator is the component that integrates new knowledge and
feedback into the playbook *without* an LLM in the loop: counter updates,
near-duplicate merging, pruning of harmful bullets, and size capping.
Everything here is pure, deterministic and unit-testable.
"""

from __future__ import annotations

import re

from .playbook import Bullet, Playbook, clean_text, utcnow_iso

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Normalise text to a set of lowercase word tokens."""
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity between the token sets of two texts (0.0–1.0)."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class Curator:
    """Applies feedback and maintenance operations to one playbook."""

    def __init__(self, playbook: Playbook) -> None:
        self.playbook = playbook

    # -- creation / editing ---------------------------------------------------
    def add_bullet(self, section: str, content: str) -> Bullet:
        bullet = Bullet(
            id=self.playbook.allocate_id(),
            section=section,
            content=content,
        )
        return self.playbook.insert(bullet)

    def refine_bullet(self, bullet_id: str, new_content: str) -> Bullet:
        """Rewrite a bullet's text (counters and history are preserved)."""
        bullet = self.playbook.get(bullet_id)
        new_content = clean_text(new_content)
        if not new_content:
            raise ValueError("bullet content must not be empty")
        bullet.content = new_content
        bullet.updated_at = utcnow_iso()
        return bullet

    # -- feedback ---------------------------------------------------------------
    def apply_feedback(self, bullet_id: str, *, helpful: bool) -> Bullet:
        """Record one vote: the bullet helped (or harmed) a task run."""
        bullet = self.playbook.get(bullet_id)
        if helpful:
            bullet.helpful_count += 1
        else:
            bullet.harmful_count += 1
        bullet.updated_at = utcnow_iso()
        return bullet

    def apply_feedback_batch(
        self, feedback: dict[str, bool]
    ) -> list[Bullet]:
        """Apply a mapping of bullet id -> helpful? in one pass."""
        return [
            self.apply_feedback(bullet_id, helpful=helpful)
            for bullet_id, helpful in feedback.items()
        ]

    # -- maintenance ------------------------------------------------------------
    def merge_duplicates(self, threshold: float = 0.8) -> list[dict]:
        """Merge near-duplicate bullets *within the same section*.

        Two bullets are near-duplicates when the Jaccard similarity of their
        token sets is >= ``threshold``.  For each duplicate cluster the
        survivor is the bullet with the most total feedback (ties: earliest
        inserted); all counters are summed into the survivor so no feedback
        evidence is lost.  Returns a report of ``{"kept", "absorbed"}`` dicts.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        bullets = self.playbook.bullets
        parent = list(range(len(bullets)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        for i, a in enumerate(bullets):
            for j in range(i + 1, len(bullets)):
                b = bullets[j]
                if a.section == b.section and jaccard(a.content, b.content) >= threshold:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(len(bullets)):
            clusters.setdefault(find(i), []).append(i)

        report: list[dict] = []
        for members in clusters.values():
            if len(members) < 2:
                continue
            # Survivor: most total feedback; ties go to the earliest inserted.
            survivor_idx = max(
                members, key=lambda i: (bullets[i].total_feedback, -i)
            )
            survivor = bullets[survivor_idx]
            group = [bullets[i] for i in members]
            absorbed = [b for b in group if b is not survivor]
            survivor.helpful_count = sum(b.helpful_count for b in group)
            survivor.harmful_count = sum(b.harmful_count for b in group)
            survivor.updated_at = utcnow_iso()
            for bullet in absorbed:
                self.playbook.remove(bullet.id)
            report.append(
                {"kept": survivor.id, "absorbed": [b.id for b in absorbed]}
            )
        return report

    def prune(self, threshold: int = -2) -> list[Bullet]:
        """Remove bullets whose net score falls *below* ``threshold``.

        The default (-2) only removes bullets that have clearly hurt more
        than helped; a fresh bullet (score 0) is never pruned by default.
        Returns the removed bullets.
        """
        doomed = [b for b in self.playbook.bullets if b.net_score < threshold]
        for bullet in doomed:
            self.playbook.remove(bullet.id)
        return doomed

    def enforce_cap(self, max_bullets: int) -> list[Bullet]:
        """Evict lowest-value bullets until the playbook fits ``max_bullets``.

        Eviction order (first out): lowest net score, then least total
        feedback, then least recently updated.  Returns evicted bullets.
        """
        if max_bullets < 0:
            raise ValueError("max_bullets must be >= 0")
        evicted: list[Bullet] = []
        while len(self.playbook) > max_bullets:
            victim = min(self.playbook.bullets, key=lambda b: b.value_key())
            self.playbook.remove(victim.id)
            evicted.append(victim)
        return evicted
