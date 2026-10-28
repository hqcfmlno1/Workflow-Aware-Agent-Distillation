# Fixed 20-task Verified diagnostic pilot

Selection frozen on 2026-10-28 for comparing the base, vanilla SFT and offline
span-anchor LoRA on the same tasks. This selection does not alter training or
reserve a new validation split.

- Official SWE-bench Verified test revision:
  `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.
- 20 distinct tasks spanning all 12 repositories; seed 42.
- One task per repository, then 8 extra slots apportioned by repository size
  using largest remainders. Sample within each repository by deterministic SHA256
  ordering, as fully specified in the JSON manifest.
- Exclude all 477 distinct tasks in the current 1,180-trajectory MetaCurate corpus.
  The complete Verified set has zero task-ID overlap with that corpus.
- Gold patches, difficulty and model results do not influence selection. Difficulty
  metadata after selection: 8 under15min, 10 at15min–1h, 2 at1–4h. These are human
  annotation categories, not estimates of agent runtime.
- Each resolved task changes the pilot score by 5 percentage points. Repository
  coverage deliberately oversamples small repositories; do not interpret this as
  an unbiased estimate of the full500 score.

Files:

- `configs/evaluation/verified20_span_anchor.ids.txt`: one instance ID per line,
  directly accepted by the existing harness `--instance-ids` option.
- `configs/evaluation/verified20_span_anchor.json`: pinned source, selection rule,
  quotas, SHA256 checksums and task metadata. No gold patch or test patch.
- Local `outputs/evaluation/verified20-span-anchor/tasks-agent-visible.json` contains
  only the existing allowed agent fields: ID, repo, base commit, version and
  problem statement.

Use the identical IDs, harness/template, dtype and decoding settings across models.
Preserve complete reasoning, tool calls, observations, patches and official grading
results. Start each checkpoint in its own output directory. Selection imposes no
new token, context, step or repetition limits.

The tasks are prepared for later benchmarking. No inference host was launched and
no existing scoring/training job was modified for this selection.
