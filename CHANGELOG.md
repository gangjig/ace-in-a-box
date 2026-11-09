# Changelog

## 0.4.0

- Hardened the Reflector against **trace-content poisoning** (thought
  injection): a trace is untrusted data, and a directive disguised as
  a lesson could previously sail through reflection into the playbook
  and be re-injected into every future prompt. Now the LLM reflector
  prompt renders steps field by field and fences observations in
  `BEGIN/END UNTRUSTED TOOL OUTPUT` delimiters with treat-as-evidence
  instructions; every `Candidate` records its `source` (`lesson`,
  `error`, `outcome`, `llm`); a new `needs_quarantine` screen flags
  candidates whose text directs moving sensitive material (keys,
  credentials, tokens) to an outside destination, marking them
  `quarantined`; and the new `Curator.add_candidate` integration gate
  refuses quarantined candidates unless a human passes
  `confirm_quarantined=True`. The screen is documented as a review
  tripwire, not a safety proof. Motivated by the thought-injection
  results in arXiv:2611.08811.

## 0.3.0

- Added a **monitor holdout** for eval sets (`ace_in_a_box.evalset`),
  so a playbook cannot quietly overfit the cases it is judged by:
  `split_cases` deterministically hash-splits an eval set into a drive
  set and a holdout set (a case's partition never changes),
  `HoldoutGuard` raises `HoldoutLeakError` if a holdout case's outcome
  is about to feed an update, and the new `ace holdout report` CLI
  compares drive vs holdout scores side by side at release time —
  warning when the drive set improves but the holdout does not — and
  appends every verdict to `holdout-ledger.jsonl`. Inspired by
  Google's stated practice of not training against its own monitors
  in the Gemini 4 Argon safety reporting, as read by Zvi Mowshowitz.

## 0.2.0

- Added an optional **task-brief header** for prompt rendering:
  `TaskBrief` (goal / verify / effort, following the prompting formula
  shared by Anthropic's Boris Cherny) can be passed to `render_prompt`
  or set via `ace render --goal/--verify/--effort`. The brief is
  prepended to the rendered playbook, counts toward the render budget,
  and is never truncated — bullets are dropped instead. Briefs are
  per-task and render-time only; they are not stored in the playbook
  file.

## 0.1.0

- Initial release: the ACE playbook loop (Generator prompt rendering,
  Reflector, Curator with feedback counters, duplicate merging,
  pruning and size caps), lossless JSON and human-editable Markdown
  persistence, budget-aware prompt rendering, and the `ace` CLI.
