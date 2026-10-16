# Changelog

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
