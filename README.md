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

## License

MIT © 2026 Gang Ji
