# Changelog

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
