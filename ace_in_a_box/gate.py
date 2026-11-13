"""Evidence-governed adoption: an EGE-style gate for playbook updates.

The Reflector proposes; the Curator integrates.  Between the two sits a
question the loop itself cannot answer: *did this candidate actually
make the agent better?*  A plausible lesson is not necessarily a good
one — it may help the case it came from while quietly degrading
everything else.  This module implements the adoption discipline of
Evidence-Governed Evolution (EGE; Lonkar, arXiv:2611.16378), adapted to
a playbook:

1. **Freeze the candidate.**  ``FrozenCandidate`` snapshots the exact
   (section, content) pairs under audit, with a fingerprint.  The text
   that gets adopted is byte-for-byte the text that was audited; the
   decision is bound to the fingerprint, so audited text cannot drift
   on its way into the playbook.
2. **Audit on fresh paired cases.**  ``run_audit`` scores the same
   fresh cases twice — once with the current playbook, once with the
   candidate playbook — and ``audit_pairs`` judges the paired scores.
   Audit cases must be fresh (never used to generate the candidate):
   pass the ids the candidate was learned from as ``prior_case_ids``
   and any overlap fails loudly.
3. **Require real improvement on the primary objective.**  Adoption
   needs a mean paired improvement of at least ``min_improvement``
   *and* a Hoeffding lower confidence bound above zero — the same
   bound family EGE builds its guarantees on.  The bound is
   deliberately conservative: with few audit cases it stays wide and
   the gate refuses.  A gate that passes on three cases is not a gate.
4. **Protect declared capabilities.**  Each declared capability's mean
   paired change must stay within ``non_inferiority_margin`` of the
   baseline.  This check is mean-based, not bound-based, on purpose:
   at realistic audit sizes a Hoeffding bound is so wide it would veto
   every adoption, including ones where capabilities held exactly
   steady.  The per-capability bounds are still computed and reported
   in the decision, so the evidence stays visible; they are simply
   not decisive.  If bound-decisive capability checks matter to you,
   audit with hundreds of cases, where the bounds tighten enough to
   be usable.
5. **Governance before canary.**  A frozen set containing quarantined
   content (see :func:`reflector.needs_quarantine`) is rejected unless
   a human confirms it, the same rule as ``Curator.add_candidate``.
   An adopted candidate then rolls out through ``CanaryRollout`` in
   bounded stages (default 10% → 50% → 100% of cases, assigned
   deterministically by case id), so a bad adoption meets a tenth of
   traffic before it meets all of it.

What this module does *not* do: it does not run your agent.  The
``runner`` callable in :func:`run_audit` is yours — it receives a
playbook and a case and returns a :class:`CaseScore`.  The gate judges
the scores it is given; it cannot verify that the runner really ran
fresh cases, that the scores are honest, or that the audit
distribution matches production.  Those are the same conditions EGE's
own guarantees are conditional on (bounded paired scores, fresh
sampling, a valid reference, declared evaluation distributions), and
here they are the caller's responsibility.  The gate makes a bad
adoption *harder*; it does not make one impossible.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable

from .playbook import Bullet, Playbook, clean_text
from .reflector import needs_quarantine

#: Verdicts returned by the gate.
VERDICT_ADOPT = "adopt"
VERDICT_REJECT = "reject"

#: Default seed for deterministic canary assignment.
DEFAULT_CANARY_SEED = "ace-canary-v1"


# ---------------------------------------------------------------------------
# Frozen candidates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrozenCandidate:
    """An immutable snapshot of the candidate bullets under audit.

    ``bullets`` is a tuple of ``(section, content)`` pairs.  The
    fingerprint is derived from those pairs on every access, so it can
    never disagree with the frozen text.  ``quarantined`` is set when
    any bullet trips :func:`reflector.needs_quarantine` (or was flagged
    by hand at freeze time); like ``Candidate``, the screen can only
    add the flag, never remove one.
    """

    bullets: tuple[tuple[str, str], ...]
    quarantined: bool = False

    def __post_init__(self) -> None:
        if not self.bullets:
            raise ValueError("cannot freeze an empty candidate set")
        cleaned: list[tuple[str, str]] = []
        for section, content in self.bullets:
            content = clean_text(str(content))
            if not content:
                raise ValueError("frozen bullet content must not be empty")
            cleaned.append((str(section), content))
        object.__setattr__(self, "bullets", tuple(cleaned))
        if any(needs_quarantine(content) for _, content in cleaned):
            object.__setattr__(self, "quarantined", True)

    @property
    def fingerprint(self) -> str:
        """SHA-256 over the frozen pairs; the audit/adoption binding."""
        payload = "\n".join(f"{section}\0{content}" for section, content in self.bullets)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_candidates(cls, candidates) -> "FrozenCandidate":
        """Freeze Reflector candidates (any objects with ``section``,
        ``content`` and optional ``quarantined`` attributes)."""
        cands = list(candidates)
        if not cands:
            raise ValueError("cannot freeze an empty candidate set")
        quarantined = any(getattr(c, "quarantined", False) for c in cands)
        return cls(
            bullets=tuple((c.section, c.content) for c in cands),
            quarantined=quarantined,
        )


def _copy_playbook(playbook: Playbook) -> Playbook:
    """A deep-enough copy: same ids, sections, contents and counters."""
    return Playbook([Bullet.from_dict(b.to_dict()) for b in playbook.bullets])


def build_candidate_playbook(base: Playbook, frozen: FrozenCandidate) -> Playbook:
    """The playbook under audit: a copy of *base* plus the frozen bullets.

    The base playbook is never mutated; frozen bullets get fresh ids.
    """
    candidate = _copy_playbook(base)
    for section, content in frozen.bullets:
        candidate.insert(
            Bullet(id=candidate.allocate_id(), section=section, content=content)
        )
    return candidate


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def _check_score(value: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} is not a number: {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{what} outside [0, 1]: {value!r}")
    return float(value)


@dataclass(frozen=True)
class CaseScore:
    """One runner result: a primary score plus declared capabilities.

    All scores are bounded in [0, 1] — the paired differences the gate
    reasons about live in [-1, 1], which is what makes the Hoeffding
    bound valid.  ``capabilities`` maps a declared capability name
    (e.g. ``"no_regression_on_refusals"``) to its score on this case.
    """

    primary: float
    capabilities: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "primary", _check_score(self.primary, "primary score"))
        checked = {
            str(name): _check_score(value, f"capability {name!r} score")
            for name, value in self.capabilities.items()
        }
        object.__setattr__(self, "capabilities", checked)


@dataclass(frozen=True)
class PairedScore:
    """Before/after scores for one audit case.

    ``capabilities`` maps each declared capability name to a
    ``(before, after)`` pair.  Every pair in an audit must declare the
    same capability names: a capability that is only sometimes measured
    cannot protect anything.
    """

    case_id: str
    before: float
    after: float
    capabilities: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.case_id):
            raise ValueError("paired score needs a non-empty case_id")
        object.__setattr__(self, "case_id", str(self.case_id))
        object.__setattr__(self, "before", _check_score(self.before, "before score"))
        object.__setattr__(self, "after", _check_score(self.after, "after score"))
        checked = {
            str(name): (
                _check_score(pair[0], f"capability {name!r} before score"),
                _check_score(pair[1], f"capability {name!r} after score"),
            )
            for name, pair in self.capabilities.items()
        }
        object.__setattr__(self, "capabilities", checked)


# ---------------------------------------------------------------------------
# Policy and decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GatePolicy:
    """The gate's acceptance thresholds (all prespecified, pre-audit).

    ``min_improvement`` is the smallest mean paired gain on the primary
    objective that counts as meaningful.  ``non_inferiority_margin`` is
    how far a declared capability's mean paired score may fall below
    the baseline before the gate rejects.  ``alpha`` is the Hoeffding
    bound's error level (the bound holds with probability >= 1-alpha
    under fresh sampling).  ``min_cases`` refuses audits too small to
    say anything.  ``canary_stages`` is the default rollout schedule
    for adopted candidates.
    """

    min_improvement: float = 0.05
    non_inferiority_margin: float = 0.05
    alpha: float = 0.05
    min_cases: int = 8
    canary_stages: tuple[float, ...] = (0.1, 0.5, 1.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_improvement <= 1.0:
            raise ValueError("min_improvement must be within [0, 1]")
        if not 0.0 <= self.non_inferiority_margin <= 1.0:
            raise ValueError("non_inferiority_margin must be within [0, 1]")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be within (0, 1)")
        if self.min_cases < 1:
            raise ValueError("min_cases must be >= 1")
        if not self.canary_stages:
            raise ValueError("canary_stages must not be empty")
        previous = -1.0
        for stage in self.canary_stages:
            if not 0.0 <= stage <= 1.0:
                raise ValueError(f"canary stage outside [0, 1]: {stage!r}")
            if stage < previous:
                raise ValueError("canary_stages must be non-decreasing")
            previous = stage


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict, with the evidence it was based on.

    ``capability_lower_bounds`` are reported for transparency but are
    not decisive (see the module docstring); the decisive capability
    check is the mean delta against the non-inferiority margin.
    """

    verdict: str
    fingerprint: str
    n: int
    primary_mean_delta: float
    primary_lower_bound: float
    capability_mean_deltas: dict[str, float]
    capability_lower_bounds: dict[str, float]
    reasons: tuple[str, ...] = ()

    @property
    def adopted(self) -> bool:
        return self.verdict == VERDICT_ADOPT

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "fingerprint": self.fingerprint,
            "n": self.n,
            "primary_mean_delta": self.primary_mean_delta,
            "primary_lower_bound": self.primary_lower_bound,
            "capability_mean_deltas": dict(self.capability_mean_deltas),
            "capability_lower_bounds": dict(self.capability_lower_bounds),
            "reasons": list(self.reasons),
        }


def _hoeffding_lower_bound(deltas: list[float], alpha: float) -> float:
    """One-sided Hoeffding lower bound on the mean of paired deltas.

    Deltas are bounded in [-1, 1] (range 2), so with probability at
    least ``1 - alpha`` the true mean is at least the returned value —
    under fresh, independent sampling.  Small ``n`` means a wide bound:
    that conservatism is the point.
    """
    n = len(deltas)
    mean = sum(deltas) / n
    width = 2.0 * math.sqrt(math.log(1.0 / alpha) / (2.0 * n))
    return round(mean - width, 10)


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

def audit_pairs(
    frozen: FrozenCandidate,
    pairs: list[PairedScore],
    policy: GatePolicy | None = None,
    *,
    prior_case_ids: object = (),
    confirm_quarantined: bool = False,
) -> GateDecision:
    """Judge frozen candidates against fresh paired audit scores.

    Raises ValueError — no decision — when the audit itself is invalid:
    no pairs, duplicate case ids, audit cases overlapping
    ``prior_case_ids`` (the audit must be fresh), or inconsistent
    declared capabilities across pairs.  A *valid* audit that fails the
    evidence bar returns a REJECT decision, not an exception: rejection
    is the gate working, not an error.
    """
    policy = policy or GatePolicy()
    pairs = list(pairs)
    if not pairs:
        raise ValueError("audit needs at least one paired score")
    ids = [p.case_id for p in pairs]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate case id(s) in audit pairs: {', '.join(dupes)}")
    prior = {str(c) for c in prior_case_ids}  # type: ignore[arg-type]
    overlap = sorted(prior & set(ids))
    if overlap:
        raise ValueError(
            "audit cases must be fresh (never used to generate the "
            f"candidate); overlap with prior cases: {', '.join(overlap)}"
        )
    capability_sets = {frozenset(p.capabilities) for p in pairs}
    if len(capability_sets) > 1:
        raise ValueError(
            "declared capabilities must be measured on every audit case; "
            f"got inconsistent sets: {sorted(map(str, capability_sets))}"
        )

    n = len(pairs)
    primary_deltas = [p.after - p.before for p in pairs]
    primary_mean = round(sum(primary_deltas) / n, 10)
    primary_bound = _hoeffding_lower_bound(primary_deltas, policy.alpha)
    capability_mean_deltas: dict[str, float] = {}
    capability_lower_bounds: dict[str, float] = {}
    for name in pairs[0].capabilities:
        deltas = [p.capabilities[name][1] - p.capabilities[name][0] for p in pairs]
        capability_mean_deltas[name] = round(sum(deltas) / n, 10)
        capability_lower_bounds[name] = _hoeffding_lower_bound(deltas, policy.alpha)

    reasons: list[str] = []
    if frozen.quarantined and not confirm_quarantined:
        reasons.append(
            "frozen candidate contains quarantined content and needs "
            "human confirmation (pass confirm_quarantined=True)"
        )
    if n < policy.min_cases:
        reasons.append(
            f"only {n} audit case(s); policy requires >= {policy.min_cases}"
        )
    if primary_mean < policy.min_improvement:
        reasons.append(
            f"primary mean delta {primary_mean:+.3f} below min_improvement "
            f"{policy.min_improvement:g}"
        )
    if primary_bound <= 0:
        reasons.append(
            f"primary Hoeffding lower bound {primary_bound:+.3f} does not "
            f"exclude zero (alpha={policy.alpha:g}, n={n})"
        )
    for name, mean_delta in capability_mean_deltas.items():
        if mean_delta < -policy.non_inferiority_margin:
            reasons.append(
                f"capability {name!r} regressed: mean delta {mean_delta:+.3f} "
                f"below -non_inferiority_margin {-policy.non_inferiority_margin:g}"
            )

    if reasons:
        verdict = VERDICT_REJECT
    else:
        verdict = VERDICT_ADOPT
        reasons.append(
            f"primary mean delta {primary_mean:+.3f} (lower bound "
            f"{primary_bound:+.3f}) with {len(capability_mean_deltas)} "
            f"capabilit(ies) within margin over {n} fresh case(s)"
        )
    return GateDecision(
        verdict=verdict,
        fingerprint=frozen.fingerprint,
        n=n,
        primary_mean_delta=primary_mean,
        primary_lower_bound=primary_bound,
        capability_mean_deltas=capability_mean_deltas,
        capability_lower_bounds=capability_lower_bounds,
        reasons=tuple(reasons),
    )


#: A runner scores one case against one playbook.  It must not mutate
#: the playbook it receives (``run_audit`` passes it copies anyway).
Runner = Callable[[Playbook, dict], CaseScore]


def run_audit(
    base: Playbook,
    frozen: FrozenCandidate,
    cases: list[dict],
    runner: Runner,
    policy: GatePolicy | None = None,
    *,
    prior_case_ids: object = (),
    confirm_quarantined: bool = False,
) -> GateDecision:
    """Run a fresh paired audit: score each case with and without the
    frozen candidate, then judge the pairs with :func:`audit_pairs`.

    Each case needs an ``id``; the runner is called once per playbook
    per case, always with a copy, so the caller's playbook is never
    mutated and the two runs see identical case dicts.  The runner's
    before/after capability names must match on every case.
    """
    candidate_playbook = build_candidate_playbook(base, frozen)
    pairs: list[PairedScore] = []
    for case in cases:
        if "id" not in case or case["id"] is None or str(case["id"]) == "":
            raise ValueError(f"audit case is missing a required 'id': {case!r}")
        case_id = str(case["id"])
        before = runner(_copy_playbook(base), case)
        after = runner(_copy_playbook(candidate_playbook), case)
        if set(before.capabilities) != set(after.capabilities):
            raise ValueError(
                f"runner returned mismatched capability names for case "
                f"{case_id!r}: before={sorted(before.capabilities)}, "
                f"after={sorted(after.capabilities)}"
            )
        pairs.append(
            PairedScore(
                case_id=case_id,
                before=before.primary,
                after=after.primary,
                capabilities={
                    name: (before.capabilities[name], after.capabilities[name])
                    for name in before.capabilities
                },
            )
        )
    return audit_pairs(
        frozen,
        pairs,
        policy,
        prior_case_ids=prior_case_ids,
        confirm_quarantined=confirm_quarantined,
    )


# ---------------------------------------------------------------------------
# Bounded canary rollout
# ---------------------------------------------------------------------------

@dataclass
class CanaryRollout:
    """Stage an adopted candidate into production a fraction at a time.

    Case ids are assigned deterministically by hashing
    ``seed:case_id`` (the same idiom as the eval-set split), so a case
    that sees the candidate playbook at one stage keeps seeing it at
    later stages as the fraction grows — assignment only ever moves
    base → candidate, never the reverse.  ``advance`` moves to the next
    stage; the caller decides when the evidence at the current stage
    justifies that.
    """

    stages: tuple[float, ...] = (0.1, 0.5, 1.0)
    seed: str = DEFAULT_CANARY_SEED
    stage: int = 0

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("canary stages must not be empty")
        previous = -1.0
        for fraction in self.stages:
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"canary stage outside [0, 1]: {fraction!r}")
            if fraction < previous:
                raise ValueError("canary stages must be non-decreasing")
            previous = fraction
        if not 0 <= self.stage < len(self.stages):
            raise ValueError(
                f"stage must be within [0, {len(self.stages) - 1}], "
                f"got {self.stage!r}"
            )

    @property
    def fraction(self) -> float:
        """Share of cases currently seeing the candidate playbook."""
        return self.stages[self.stage]

    @property
    def is_full(self) -> bool:
        return self.stage == len(self.stages) - 1 and self.fraction >= 1.0

    def uses_candidate(self, case_id: str) -> bool:
        bucket = int(
            hashlib.sha256(f"{self.seed}:{case_id}".encode("utf-8")).hexdigest(), 16
        ) % 100
        return bucket < self.fraction * 100

    def playbook_for(
        self, case_id: str, base: Playbook, candidate: Playbook
    ) -> Playbook:
        """The playbook this case should run with at the current stage."""
        return candidate if self.uses_candidate(case_id) else base

    def advance(self) -> float:
        """Move to the next rollout stage; returns the new fraction."""
        if self.stage >= len(self.stages) - 1:
            raise ValueError("canary is already at its final stage")
        self.stage += 1
        return self.fraction
