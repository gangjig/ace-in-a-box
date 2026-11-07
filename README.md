# ace-in-a-box

**A mini self-evolving playbook framework for LLM agents — the ACE (Agentic Context Engineering) pattern in pure Python, zero dependencies.**

Instead of fine-tuning a model or stuffing ever more instructions into a
static prompt, ACE treats the agent's *context* as the thing that learns:
a structured **playbook** of short bullet entries — strategies, domain
facts, failure modes, code snippets — that evolves after every task run.
Bullets that help accumulate positive feedback; bullets that hurt get
pruned; near-duplicates get merged. The context stays compact, current,
and inspectable by humans, because it is just data.

## The ACE loop

ACE (from the paper *Agentic Context Engineering*, Zhang et al., 2025)
splits the system into three roles, all implemented here:

```
        ┌────────────┐   trace    ┌────────────┐  candidates  ┌────────────┐
        │ Generator  │ ─────────▶ │ Reflector  │ ───────────▶ │  Curator   │
        │ (your agent│            │ distills   │              │ integrates │
        │  + prompt) │ ◀───────── │  lessons   │              │  feedback  │
        └────────────┘  playbook  └────────────┘              └────────────┘
              ▲ rendered prompt                                     │
              └─────────────────────────────────────────────────────┘
```

- **Generator** — your agent. It receives the playbook rendered into its
  prompt (`render_prompt`) and produces an execution trace.
- **Reflector** (`Reflector`) — reads a trace + outcome and proposes
  candidate bullets. Ships with a deterministic rule-based extractor
  (errors → failure modes, recorded lessons → strategies) and an optional
  pluggable LLM callable for richer distillation.
- **Curator** (`Curator`) — the deterministic bookkeeper: applies
  helpful/harmful feedback, refines bullet text, merges near-duplicates
  (token-set Jaccard similarity), prunes bullets whose net score falls
  below a threshold, and caps the playbook size by evicting the
  lowest-value bullets.

## Quickstart

```python
from ace_in_a_box import Curator, Playbook, Reflector, render_prompt

playbook = Playbook()
curator = Curator(playbook)

# Seed a bullet (or let the Reflector propose them).
b = curator.add_bullet("strategies", "Run the test suite before committing")

# The Generator runs with the playbook in its prompt:
prompt_context = render_prompt(playbook, max_tokens=2000)

# ... the agent runs, producing a trace and an outcome ...

trace = [
    {"action": "edit_file", "observation": "patched parser"},
    {"action": "run_tests", "error": "2 failed: test_tokenize"},
    {"action": "fix", "lesson": "Tokenizer must handle empty input"},
]
outcome = {"success": False}

# The Reflector proposes new bullets from the trace:
for candidate in Reflector().reflect(trace, outcome):
    curator.add_bullet(candidate.section, candidate.content)

# Feedback from the run updates bullet counters:
curator.apply_feedback(b.id, helpful=True)

# Periodic maintenance keeps the playbook healthy:
curator.merge_duplicates(threshold=0.8)
curator.prune(threshold=-2)        # drop bullets that hurt more than help
curator.enforce_cap(max_bullets=200)
```

Persistence is one call each way, in lossless JSON or human-editable
Markdown:

```python
from ace_in_a_box import load_playbook, save_playbook

save_playbook(playbook, "playbook.json")   # or "playbook.md"
playbook = load_playbook("playbook.json")
```

## Task briefs

A playbook is standing knowledge; a **task brief** says what *this*
run is for. Pass a `TaskBrief` to `render_prompt` and it is prepended
to the rendered playbook as a header:

```python
from ace_in_a_box import TaskBrief, render_prompt

brief = TaskBrief(
    goal="Fix the tokenizer crash on empty input",
    verify="the full test suite passes",
    effort="a focused hour; stop and report if blocked",
)
prompt_context = render_prompt(playbook, brief=brief, max_tokens=2000)
```

The brief format follows the prompting formula shared by Anthropic's
Boris Cherny — say what to do, how to verify it, and how much effort to
spend — as reported in Zvi Mowshowitz's AI #190
(https://thezvi.substack.com/p/ai-190-scion-of-the-times). Only the
goal is required; `verify` and `effort` lines are omitted when not
given.

A brief is per-task and render-time only — it is **not** stored in the
playbook file. It counts toward the render budget and is never dropped
or truncated; when space runs out, playbook bullets are dropped
instead, exactly as they are without a brief.

From the CLI:

```console
$ ace render playbook.json --goal "Fix the tokenizer crash" \
      --verify "the full test suite passes" --effort "a focused hour"
```

(`--verify` and `--effort` require `--goal`.)

## Monitor holdout: don't train against your own ruler

A playbook that is always updated against the same eval set quietly
overfits it: the score on those cases climbs while real performance
stands still — the ruler goes dull, and nothing in the drive-set score
tells you. The countermeasure is a **monitor holdout**, the same
discipline Google states for its chain-of-thought monitors in the
Gemini 4 Argon safety reporting — never train against the monitor —
as covered in Zvi Mowshowitz's page-by-page read of the safety report
(https://thezvi.substack.com/p/gemini-4-argon-the-model-and-safety).

`ace_in_a_box.evalset` splits an eval set (JSONL, one case object with
an `id` per line; other fields pass through to your runner) into a
**drive** set and a **holdout** set by hashing `seed:case_id`. The
split is deterministic, order-independent, and stable as the set
grows: a case can never migrate between sets, so the holdout stays a
clean measurement. Holdout outcomes must never feed the
Reflector/Curator loop — a `HoldoutGuard` makes the common accident
fail loudly at the call site:

```python
from ace_in_a_box import HoldoutGuard, load_cases, split_cases

split = split_cases(load_cases("eval.jsonl"))      # 50/50 by default
guard = HoldoutGuard(split)
for case_id, trace, outcome in run_my_eval(split.drive):
    guard.check(case_id)   # raises HoldoutLeakError for holdout ids
    reflector.reflect(trace, outcome)
```

The guard is a call-site check, not a sandbox: ace-in-a-box does not
run your tasks, so it cannot intercept code that never calls it. It
catches the accidental leak, not a determined bypass.

At release time, score both sets (scores JSONL: `{"id", "score",
"run"}` — use `run` labels like `before`/`after` for an update) and
compare them side by side:

```console
$ ace holdout report --cases eval.jsonl --scores scores.jsonl
eval set: 20 case(s) — drive 8, holdout 12 (seed 'ace-holdout-v1', fraction 0.5)
fingerprint: 37935f94193e48ab…
run 'before': drive 0.500 (n=8)  holdout 0.500 (n=12)  gap +0.000
run 'after':  drive 0.900 (n=8)  holdout 0.520 (n=12)  gap +0.380
Δdrive +0.400  Δholdout +0.020
WARNING: drive/holdout gap 0.380 exceeds --max-gap 0.1 — the playbook does
  much better on cases it learns from than on cases it never sees
```

If the drive set improved but the holdout did not, the report prints
an overfitting warning ("ruler going dull"); if both improved, the
holdout confirms the gain. Every report appends one line to
`holdout-ledger.jsonl` (eval-set fingerprint, per-run means, deltas,
verdict), so a slowly dulling ruler shows up as history, not as a
surprise at release time. The report gives means and gaps only — no
significance tests; at eval-set sizes this small they would be false
precision.

## CLI

The `ace` command (or `python -m ace_in_a_box`) manages a playbook file
directly — format is chosen by suffix (`.json` / `.md`):

```console
$ ace init playbook.json
$ ace add playbook.json --section strategies \
      --content "Run the test suite before committing"
b-0001
$ ace feedback playbook.json b-0001 --helpful
b-0001: helpful=1 harmful=0 net=+1
$ ace show playbook.json          # print as Markdown
$ ace prune playbook.json --threshold -2 --max-bullets 100 --merge-threshold 0.8
$ ace render playbook.json --max-chars 4000   # prompt-ready text, budget-aware
$ ace holdout report --cases eval.jsonl --scores scores.jsonl
                                              # release gate: drive vs holdout
```

## Installation

```console
$ pip install -e .          # from a checkout
$ pip install -e ".[dev]"   # plus pytest, for running the tests
```

Requires Python ≥ 3.10. No runtime dependencies.

## Running the tests

```console
$ python -m pytest
```

The suite (curator ops, similarity merging, prune/eviction, Markdown +
JSON round-trips, budget rendering, reflector with a fake LLM, CLI smoke
tests) runs fully offline.

## Design notes

- **Value ordering.** A bullet's value is `(net score, total feedback,
  recency)`. Eviction and budget rendering both drop the lowest-value
  bullets first, so an untested bullet is cheaper to lose than a
  battle-tested one, and a harmful bullet goes before either.
- **Merging preserves evidence.** When two bullets merge, the survivor
  keeps the *sum* of both counters — merging never erases feedback.
- **Markdown is a first-class format.** Following ACE, the playbook is
  meant to be read and edited by people. The Markdown format keeps ids,
  sections, contents and counters (timestamps live only in JSON).

## Limitations (honest ones)

- **This is the structured-context pattern, not model training.** No
  weights change; the "learning" lives entirely in the playbook file.
  That is the point of ACE, but it also means the playbook can only
  capture what fits in a prompt.
- **The Generator is yours.** ace-in-a-box manages the playbook, the
  reflection and the curation — it does not run agents or call models
  for you. The optional LLM hook for the Reflector is a plain callable;
  wiring it to a real model is a few lines on your side.
- **Similarity is lexical.** Near-duplicate detection uses token-set
  Jaccard similarity: fast, deterministic and dependency-free, but it
  will miss paraphrases that share few words. Swap in embeddings behind
  the same interface if you need semantic merging.
- **Feedback quality is your problem.** Counters are only as good as the
  signal that increments them; garbage votes produce a confidently
  wrong playbook. Human review of Reflector candidates is recommended
  before integration in serious use.

## Build your own

This repository is small enough to regenerate from a prompt. Paste the
following into an LLM and it should produce a closely similar project:

> Write a small, zero-dependency Python framework that implements the
> ACE (Agentic Context Engineering) pattern: instead of fine-tuning a
> model, let an agent's context be the thing that learns, as a
> structured playbook of short bullet entries (strategies, domain facts,
> failure modes, code snippets) that evolves after every task run. The
> system has three roles: a generator, the user's own agent, which runs
> with the playbook rendered into its prompt and produces a trace; a
> reflector, which distills a trace and its outcome into candidate
> lessons; and a curator, which does the deterministic bookkeeping:
> applying helpful and harmful feedback to bullets, refining their text,
> merging near-duplicates, pruning bullets whose net score falls below a
> threshold, and capping the playbook size. The playbook itself stays
> plain, inspectable data, so a human can read exactly what the agent
> has learned and why. Ship a rule-based reflector that works with no
> model call at all, an optional pluggable LLM callable for richer
> distillation, and a monitor-holdout discipline so the playbook is
> never scored against the same ruler it trained on.

## License

MIT © 2026 Gang Ji
