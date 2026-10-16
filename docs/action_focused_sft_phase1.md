# Action-focused SFT — Phase 1

Phase 1 adapts an already capable student using teacher reasoning, tool calls,
Bash commands, and terminal submission. Conditional action supervision targets
both command syntax and action selection given the preceding teacher history.
The [loss design](action_focused_sft_loss_design.md) specifies SmartAD-style
weighted-token CE with reasoning/content/action/submit weights **1 / 0.75 / 1.5 / 2**.

The local machine prepares data on CPU and stores the tokenizer/chat template;
fine-tuning runs remotely, provisionally on Kaggle. The preparation pipeline does
not load model weights, initialize a model, use GPUs, train/evaluate the student,
or execute teacher commands. The older SWE-agent collector remains unchanged.

## Run the pilot pipeline

```bash
uv sync --extra data
uv run python scripts/data/prepare_action_focused_sft.py
uv run python scripts/data/prepare_action_focused_sft.py --offline
uv run python scripts/data/prepare_action_focused_sft.py --verify
```

The `data` extra provides Transformers/tokenizers, Jinja2, and jsonschema; the
`train` extra is unnecessary. Offline mode requires complete caches. Outputs with
an existing `report.json` are not overwritten; change `local_paths.processed` in
the configuration for a new build. An incomplete run can reuse its cache.

The `nvidia_mini_swe_agent_v1_2_action_focused_sft.json` manifest has a separate
`phase1` selection: 250 train and 25 originally reserved tasks per category
(`bug-fix`, `feature-request`, `enhancement`, `other`). The current validation set
retains 50 tasks using seed 42, with category counts 13/13/12/12 in that order.
The other 50 reserved tasks were returned to the training pool. The 1,000-task
pilot training set is unchanged. There is no internal test split; the configured
final benchmark is SWE-bench Verified's 500 tasks.

Selection uses Qwen3.8-27B Thinking, `resolved=1`, the audited NVIDIA revision, and
all 44 source shards. Sort repositories/tasks/attempts before seeded shuffling;
sample repositories round robin, reserve held-out tasks first, and select one
trajectory per task. Sort and shuffle reserved task keys within each category
with seed 42 to retain validation and release the remainder. A quota shortage
stops the build instead of changing the distribution.

Read all 494 base and 415 style OmniCode instances; six source-file checksums must
match the audit. Canonical task keys use lowercase repository and PR number.
Style variants use `original_instance_id`/`pull_number` to avoid confusing variant
numbers with PRs. This does not detect different PRs addressing the same issue.

## Pilot artifacts

Files under `data/processed/mini_swe_agent_v1_2_phase1/` are ignored by Git:

| File | Contents |
| --- | --- |
| train.jsonl, validation.jsonl | Normalized messages/tools, text, token counts, checks, provenance, action candidates |
| task_selection_manifest.json | Config/seed, task IDs, benchmark hashes, tokenizer commit/template/file hashes |
| excluded.jsonl | Pre-sampling exclusions and reasons |
| errors.jsonl | Failed normalization/serialization candidates, replaced within category |
| checks.jsonl | Schema-invalid calls and unverified cases |
| task_pool.jsonl | All benchmark-disjoint attempts, grouped by task partition |
| report.json | Category/repository/source/language/operation coverage, lengths, checks, artifact hashes |
| verification_report.json | Offline source/template/token/span and pool-partition checks |
| failure.json | Quota-shortage details, if any |

Every subsequent phase and dataset extension must exclude all tasks with
`partition=validation`, including unselected, unsuccessful, or differently
categorized attempts. Pool membership alone is not training eligibility: filter
teacher/category/resolution and validate conversations before use.

Read provenance and category/teacher metadata, without reading NVIDIA reference
or model patches. Preserve content, reasoning, all tool calls, and observations.
Decode the tools and arguments JSON layers once without additional command
unescaping. Caches retain source messages. Normalized records retain reasoning
even if the template omits it from history. The official Qwen3.5-4B template uses
`add_generation_prompt=False`; preparation adds no truncation, packing, or labels.

Parseable but schema-invalid calls are retained with an invalid check because
teachers can make mistakes before recovering. Unnormalizable JSON or invalid
observation ordering rejects a trajectory. Observations without IDs receive only
positional matching and an unverified status. Terminal submission without a final
observation is allowed. `resolved=1` does not establish command validity.

Action candidates contain Python character offsets `[start,end)` in rendered text
and message/call indices. Marker rendering distinguishes actual calls from tags
inside reasoning, content, and observations. Removing markers must reproduce the
official render exactly. A non-string final argument has a null span and an
unverified status. Candidates are not token labels or evidence of executability.

## Student protocol diagnostics

Compare base and adapter outputs under the same harness, prompt, tool schema,
image/dependency state, task split, generation settings, and budget. Record model,
adapter, parser revisions, and image digests. These metrics have not been measured
by the preparation pipeline and are secondary to task success for the first run.

**Tool-call validity:** output must pass the Qwen harness parser, tool-name checks,
and JSON schema on the first attempt, without retry, JSON repair, or fallback.
The denominator includes all generations requested to produce an action; missing
or unparseable calls are invalid. Report call-level and turn-level rates, with a
multi-call turn passing only when every call is valid. Report legitimate harness
final responses separately rather than adding them to the action denominator.

**Command availability:** each program or builtin in valid Bash calls must be
confirmed in the matching harness shell/PATH/cwd/task/setup state at that moment.
Do not infer availability from the local machine's PATH or `resolved=1`. Compound
commands require checking each invocation; one confirmed missing invocation makes
the call unavailable. Report `verified_available`, `verified_unavailable`, and
`unverified` counts, availability among verified calls, and verification coverage.

Dynamic commands, substitutions, unseen scripts, changing PATH, and unavailable
dependency/environment state remain unverified. Availability does not imply a zero
exit code or task correctness. Reported operation classes are first-program
heuristics, not availability measurements. Dataset verification does not establish
student performance.

## Evaluation policy and historical data

The first experiment uses fixed weights and trains for one epoch over an explicit
context-fitting subset of the 4,000 rule-selected trajectories, then evaluates the
final checkpoint on all 500 tasks of
`princeton-nlp/SWE-bench_Verified`, revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Teacher-trajectory validation CE is an
optional diagnostic, not the checkpoint-selection criterion. Compare base and
adapter with matched evaluation conditions and report resolved rate and resource
use. Save full training checkpoints for reliable resume. No dataset expansion or
second epoch is currently planned.

The `test_evaluation` configuration records the dataset, revision, split, and
audit. Keep the first test independent of weight/checkpoint selection. Repeated
benchmark-driven tuning makes Verified a development benchmark and must be
reported accordingly or followed by a separate untouched test.

The audit in `reports/swebench_verified_overlap_2026-10-12/` found no canonical
repository-plus-PR overlap with 107,267 source trajectories. It does not establish
absence of semantic duplicates or shared issues.

The existing 50 validation tasks remain held out. All attempts from the former
50-task internal test were returned to the pool. Historical `test.jsonl`,
manifest/report, and success exports are archived under
`data/interim/archive_internal_test_2026-10-12/`, outside active paths. Processed
pilot records are complete trajectories; next-action examples and labels have not
been built. Active success exports use `train_pool`, `validation`, and
`excluded_omnicode` partitions.

## Full-corpus template conversion

The separate CPU-only `scripts/data/convert_success_trajectories.py` pipeline
renders every success-export row using the same pinned official Qwen3.5-4B
tokenizer as the pilot. Sharded gzip JSONL outputs are under
`data/processed/nvidia_mini_swe_agent_v1_2_success_qwen35_4b/`. It preserves task
partitions and all attempts without changing the 1,000+50 pilot selection.
Normalization failures remain explicit error records with original payloads;
there is no silent replacement or dropping. Loss masks and SFT training labels
remain a trainer-specific step. See [data instructions](../data/README.md) for
build, replay verification, and streaming-read commands.

## Rule-selected SFT baseline

The current baseline selects 4,000 distinct train-pool tasks with multiple
successful attempts, keeping one original Qwen-rendered trajectory per task.
`scripts/data/select_rule_trajectories.py` ranks candidates lexicographically by
confirmed historical shell usage errors, consecutive duplicate action/observation
blocks, serialized token count, tool-call count, and trajectory ID. It excludes
schema-invalid calls and keeps held-out tasks out of sampling. Category quotas
are proportional to eligible tasks; repository sampling is seeded round robin.

Outputs, diagnostics, remaining winners, and verification are under
`data/processed/mini_swe_agent_rule_baseline_4000/`; see
[data instructions](../data/README.md) for details. No pilot or validation record
is changed. The [loss design](action_focused_sft_loss_design.md) specifies
weighted-token CE, masking, per-trajectory reduction, the one-epoch evaluation
policy, and checkpointing requirements. CPU token masks and a remote weighted-loss
pilot are implemented; annotation of all 4,000 trajectories has been verified.
The original precision probe stopped on OOM at 8,044 tokens. Optimized QLoRA
subsequently completed two finite optimizer steps at 16,320 tokens on both one
T4 and a partitioned pair of T4s, before OOM at 32,766 tokens. Further internal
Gated DeltaNet compute blocking retains complete histories and attached state
gradients; see the current results in the [Kaggle instructions](../cloud/kaggle/README.md).
Baseline SFT has not started.
Rules are a quality/complexity proxy, not student NLL alignment.

## Initial training context policy

Retain complete trajectories that fit the measured remote training limit. Preserve
longer originals separately for future processing. Do not truncate, create sliding
windows, or sample replacement tasks for the initial run. The selected count must
be made explicit before training; a native model context limit does not establish
training feasibility on a T4.

| Context cap (tokens) | Complete trajectories among the selected 4,000 |
| --- | ---: |
| 8,192 | 4 |
| 16,384 | 220 |
| 32,768 | 1,416 |
| 65,536 | 3,145 |
| 131,072 | 3,924 |
| 262,144 | 4,000 |

These are length counts, not measured hardware limits. Next-step examples retain
the preceding history when built faithfully, so later steps can still exceed the
training limit. Splitting targets alone does not solve that input-length problem.
The earlier 5,225-token probe is superseded by full-history QLoRA optimization.
Sampled lengths do not establish a universal maximum context or an effective
training count. No replacement sampling or shortened-context baseline is active.
