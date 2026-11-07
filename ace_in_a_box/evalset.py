"""Monitor holdout: keep part of an eval set out of the learning loop.

If a playbook is always updated against the same eval set, the eval set
itself becomes part of what is being optimized — the ruler goes dull,
and a rising score stops meaning the playbook got better.  The fix is
the one Google states for its own chain-of-thought monitors in the
Gemini 4 Argon safety reporting (as read by Zvi Mowshowitz): never
train against the monitor.  Here that becomes three pieces:

- :func:`split_cases` — a deterministic hash split of an eval set into
  a *drive* set (whose outcomes may feed Reflector/Curator updates)
  and a *holdout* set (whose outcomes never may).  A case's partition
  is a pure function of its id, so a case can never migrate between
  sets when the eval set is re-run, reordered or extended.
- :class:`HoldoutGuard` — a call-site check to run before feeding an
  outcome into the evolution loop.  It raises :class:`HoldoutLeakError`
  for holdout cases.  The framework does not run tasks, so the guard
  cannot *force* anything: it exists so the common accident — feeding
  the holdout in by mistake — fails loudly at the first call site
  instead of silently dulling the ruler.
- :func:`build_report` (CLI: ``ace holdout report``) — a release gate
  that scores both sets side by side.  If the drive set improved but
  the holdout did not, the playbook is probably overfitting the drive
  set; every report is appended to a ledger so the history of the gap
  stays visible.

Eval sets and score files are JSONL.  A case is any JSON object with a
required ``id``; all other fields are passed through untouched for the
caller's own runner.  A score line is ``{"id": ..., "score": 0..1}``
with an optional ``run`` label (e.g. ``before`` / ``after``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SEED = "ace-holdout-v1"

# A delta at or below this is "flat" for verdict purposes; above it the
# change counts as real movement rather than scoring noise.
_EPS = 0.01

OVERFIT_WARNING = (
    "WARNING: drive set improved but holdout did not — the playbook "
    "may be overfitting the drive set (ruler going dull)"
)
CONFIRMED_OK = "OK: holdout confirms the drive-set gain"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: str | Path, kind: str) -> list[dict]:
    rows: list[dict] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{kind} line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{kind} line {lineno}: expected a JSON object")
        rows.append(row)
    return rows


def load_cases(path: str | Path) -> list[dict]:
    """Load an eval set from a JSONL file (one case object per line)."""
    return _load_jsonl(path, "cases")


def load_scores(path: str | Path) -> list[dict]:
    """Load per-case scores from a JSONL file (see module docstring)."""
    return _load_jsonl(path, "scores")


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------

def _case_id(case: dict) -> str:
    if "id" not in case or case["id"] is None or str(case["id"]) == "":
        raise ValueError(f"eval case is missing a required 'id': {case!r}")
    return str(case["id"])


def _bucket(seed: str, case_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def _fingerprint(seed: str, holdout_fraction: float, case_ids: list[str]) -> str:
    payload = "\n".join(
        [f"seed={seed}", f"fraction={holdout_fraction!r}"]
        + sorted(case_ids)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvalSplit:
    """The result of splitting an eval set into drive and holdout sets.

    ``fingerprint`` covers the seed, the fraction and the sorted case
    ids: change any of them and reports/ledger entries show a different
    fingerprint, so a swapped ruler is visible at a glance.  (Changing
    the fraction *is* changing the ruler — boundary cases move — so it
    is part of the split's identity.)
    """

    drive: list[dict]
    holdout: list[dict]
    fingerprint: str
    seed: str = DEFAULT_SEED
    holdout_fraction: float = 0.5

    @property
    def drive_ids(self) -> frozenset[str]:
        return frozenset(_case_id(c) for c in self.drive)

    @property
    def holdout_ids(self) -> frozenset[str]:
        return frozenset(_case_id(c) for c in self.holdout)


def split_cases(
    cases: list[dict],
    seed: str = DEFAULT_SEED,
    holdout_fraction: float = 0.5,
) -> EvalSplit:
    """Split *cases* into drive and holdout sets, deterministically.

    A case lands in the holdout set iff
    ``sha256(f"{seed}:{case_id}") % 100 < holdout_fraction * 100``.
    Because the partition depends only on the (seed, id) pair — never
    on position or a random draw — a case can never migrate between
    sets across runs, reorderings or extensions of the eval set, and
    raising the fraction only ever moves cases drive → holdout.
    """
    if not cases:
        raise ValueError("cannot split an empty eval set")
    if not 0.0 <= holdout_fraction <= 1.0:
        raise ValueError(
            f"holdout_fraction must be within [0, 1], got {holdout_fraction!r}"
        )
    ids = [_case_id(c) for c in cases]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate case id(s) in eval set: {', '.join(dupes)}")
    threshold = holdout_fraction * 100
    drive: list[dict] = []
    holdout: list[dict] = []
    for case, case_id in zip(cases, ids):
        (holdout if _bucket(seed, case_id) < threshold else drive).append(case)
    return EvalSplit(
        drive=drive,
        holdout=holdout,
        fingerprint=_fingerprint(seed, holdout_fraction, ids),
        seed=seed,
        holdout_fraction=holdout_fraction,
    )


# ---------------------------------------------------------------------------
# Update guard
# ---------------------------------------------------------------------------

class HoldoutLeakError(Exception):
    """Raised when a holdout case's outcome is about to feed an update."""


class HoldoutGuard:
    """Call-site guard: check a case id before its outcome teaches.

    Usage in the caller's own evolution loop::

        guard = HoldoutGuard(split_cases(cases))
        for case_id, outcome in run_results:
            guard.check(case_id)          # raises for holdout cases
            reflector.reflect(trace, outcome)

    Cases that are not in the eval set at all (ordinary production
    traces) pass: the guard only protects the holdout set.
    """

    def __init__(self, split: EvalSplit):
        self._split = split
        self._holdout_ids = split.holdout_ids

    @property
    def split(self) -> EvalSplit:
        return self._split

    def is_holdout(self, case_id: str) -> bool:
        return str(case_id) in self._holdout_ids

    def check(self, case_id: str) -> bool:
        """Return True if *case_id* may feed updates; raise otherwise."""
        if self.is_holdout(case_id):
            raise HoldoutLeakError(
                f"case {case_id!r} is in the monitor holdout set; its "
                "outcome must not feed playbook updates"
            )
        return True


# ---------------------------------------------------------------------------
# Release-gate report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunMeans:
    """Mean scores for one run label, per partition."""

    run: str
    drive_mean: float
    holdout_mean: float
    drive_n: int
    holdout_n: int

    @property
    def gap(self) -> float:
        return self.drive_mean - self.holdout_mean


@dataclass(frozen=True)
class HoldoutReport:
    """A side-by-side drive/holdout score report with a verdict."""

    split: EvalSplit
    runs: list[RunMeans]
    max_gap: float
    delta_drive: float | None = None
    delta_holdout: float | None = None
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def latest(self) -> RunMeans:
        return self.runs[-1]

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.split.fingerprint,
            "seed": self.split.seed,
            "holdout_fraction": self.split.holdout_fraction,
            "n_cases": len(self.split.drive) + len(self.split.holdout),
            "n_drive": len(self.split.drive),
            "n_holdout": len(self.split.holdout),
            "runs": [
                {
                    "run": r.run,
                    "drive_mean": r.drive_mean,
                    "holdout_mean": r.holdout_mean,
                    "drive_n": r.drive_n,
                    "holdout_n": r.holdout_n,
                    "gap": r.gap,
                }
                for r in self.runs
            ],
            "delta_drive": self.delta_drive,
            "delta_holdout": self.delta_holdout,
            "verdict": self.verdict,
        }

    def ledger_entry(self) -> dict:
        """This report as one ledger line (timestamped, UTC)."""
        entry = {"ts": datetime.now(timezone.utc).isoformat()}
        entry.update(self.to_dict())
        return entry


def build_report(
    split: EvalSplit,
    scores: list[dict],
    max_gap: float = 0.1,
) -> HoldoutReport:
    """Score *scores* against *split* and judge the drive/holdout gap.

    Score rows are grouped by their ``run`` label (unlabeled rows form
    a single run).  Deltas compare the first run against the last.
    Raises ValueError — no report — if either partition ends up with
    no scored cases or a score references an unknown case id: a gate
    that can be satisfied by an empty ruler is not a gate.
    """
    if not split.drive or not split.holdout:
        raise ValueError(
            "eval split has an empty partition "
            f"(drive={len(split.drive)}, holdout={len(split.holdout)}); "
            "adjust --fraction or add cases"
        )
    if not scores:
        raise ValueError("no scores provided")

    partition = {cid: "drive" for cid in split.drive_ids}
    partition.update({cid: "holdout" for cid in split.holdout_ids})

    run_order: list[str] = []
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in scores:
        if "id" not in row or "score" not in row:
            raise ValueError(f"score row needs 'id' and 'score': {row!r}")
        case_id = str(row["id"])
        if case_id not in partition:
            raise ValueError(f"score references unknown case id: {case_id!r}")
        score = row["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"score for case {case_id!r} is not a number: {score!r}")
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"score for case {case_id!r} outside [0, 1]: {score!r}")
        run = str(row.get("run") or "")
        if run not in grouped:
            grouped[run] = {"drive": [], "holdout": []}
            run_order.append(run)
        grouped[run][partition[case_id]].append(float(score))

    runs: list[RunMeans] = []
    for run in run_order:
        drive_scores = grouped[run]["drive"]
        holdout_scores = grouped[run]["holdout"]
        if not drive_scores or not holdout_scores:
            raise ValueError(
                f"run {run!r} has no scored cases in one partition "
                f"(drive={len(drive_scores)}, holdout={len(holdout_scores)}); "
                "refusing to report on an unscored set"
            )
        runs.append(
            RunMeans(
                run=run,
                drive_mean=round(sum(drive_scores) / len(drive_scores), 10),
                holdout_mean=round(sum(holdout_scores) / len(holdout_scores), 10),
                drive_n=len(drive_scores),
                holdout_n=len(holdout_scores),
            )
        )

    delta_drive: float | None = None
    delta_holdout: float | None = None
    ok_verdict: str | None = None
    warnings: list[str] = []
    if len(runs) >= 2:
        delta_drive = round(runs[-1].drive_mean - runs[0].drive_mean, 10)
        delta_holdout = round(runs[-1].holdout_mean - runs[0].holdout_mean, 10)
        drive_up = delta_drive > _EPS
        holdout_up = delta_holdout > _EPS
        if drive_up and not holdout_up:
            warnings.append(OVERFIT_WARNING)
        elif drive_up and holdout_up:
            ok_verdict = CONFIRMED_OK
        elif holdout_up:
            ok_verdict = "OK: holdout improved while the drive set held steady"
        elif delta_drive < -_EPS or delta_holdout < -_EPS:
            warnings.append(
                "WARNING: drive/holdout scores regressed — the update "
                "made things worse, not just duller"
            )
        else:
            ok_verdict = "OK: no measurable change on either set"

    gap = runs[-1].gap
    if gap > max_gap:
        warnings.append(
            f"WARNING: drive/holdout gap {gap:.3f} exceeds --max-gap "
            f"{max_gap:g} — the playbook does much better on cases it "
            "learns from than on cases it never sees"
        )
    if ok_verdict is None and not warnings:
        ok_verdict = (
            f"OK: drive/holdout gap {gap:.3f} within --max-gap {max_gap:g} "
            "(single run — score before/after an update to watch the ruler)"
        )
    verdict = warnings[0] if warnings else (ok_verdict or "")
    return HoldoutReport(
        split=split,
        runs=runs,
        max_gap=max_gap,
        delta_drive=delta_drive,
        delta_holdout=delta_holdout,
        verdict=verdict,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Ledger (append-only history of report verdicts)
# ---------------------------------------------------------------------------

def append_ledger(path: str | Path, entry: dict) -> Path:
    """Append *entry* as one JSON line to the ledger at *path*.

    Written atomically (full content via a temp file + ``os.replace``)
    in the style of the playbook persistence layer, so a crash mid-write
    can never leave a torn ledger.
    """
    path = Path(path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(existing + line, encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_ledger(path: str | Path) -> list[dict]:
    """Read all ledger entries at *path* (empty list if none exist)."""
    path = Path(path)
    if not path.exists():
        return []
    return _load_jsonl(path, "ledger")
