# Offline Span-Weighted LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a fresh LoRA with offline action-informed reasoning weights and a frozen-base target-token anchor, preserving the vanilla corpus and comparison protocol.

**Architecture:** CPU export establishes exact rendered span/action provenance. One frozen base computes resumable scoring and reference-logp sidecars; training then uses only one resident base plus LoRA and chunked token loss. New modules/profile preserve historical objectives and receipts.

**Tech Stack:** Existing PyTorch/Transformers/PEFT BF16 trainer, FLA, NumPy memory maps, JSONL manifests, Hugging Face epoch uploads.

**Spec:** [Experimental design and loss](../../research/offline_span_lora_design_2026-10-27.md).

## Global Constraints

- Base/revision, corpus index hash, 1.180 attempts, masks and shuffle match the spec.
- Rank 32, alpha 64, dropout 0.05; LR 1e-4, warmup 0.05, two epochs, accumulation 4.
- Span floor 0.1, dead zone 0.05 nats, maximum 24 merged spans/turn, beta 0.03.
- Reference is frozen raw base; score offline once; no teacher calls or executed actions.
- Adaptive CE weights apply only to rendered inner reasoning. Anchor weights are 1 on whole rationale sections (including tags)/envelope and 1e-6 on other supervised response tokens; action and submit remain distinct.
- Preserve trajectory-mean reduction; no hidden holdout, truncation or altered historical artifacts.
- Local latest recovery plus epoch endpoints; HF uploads at epoch endpoints only.
- This is a plan, not authorization to start a GPU instance or spend on scoring/training.

## Review Focus

- Template-omitted reasoning must never acquire fabricated supervised weights (Task 1).
- Multiple tool calls and terminal submit must retain their complete serialized targets (Task 1).
- Resuming a partly written or incompatible sidecar must reject stale/corrupt rows (Task 2).
- Dropout/mixed precision must not invalidate reference identity tests or destabilize exp (Task 3).
- Failed upload/disk pressure must preserve an epoch endpoint and its exportable logs (Task 4).

---

### Task 1: Exact rendered span/action provenance

**Files:** Create `src/workflow_distillation/lora_fp16/span_data.py`, `scripts/data/prepare_span_lora.py`, `tests/test_span_lora_data.py`.

**Interfaces:** `prepare_turns(messages: list[dict], tools: list[dict], tokenizer: object, expected_ids: list[int], groups: list[int]) -> list[dict]`. Each turn record contains message index, rendered prefix/text offsets, disjoint reasoning spans, action target IDs, group IDs and scoring eligibility. `span_weights(deltas: list[float], floor: float=0.1, dead_zone: float=0.05) -> list[float]` implements spec ranks/spread.

- [ ] Write failing tests: omitted earlier reasoning produces no target span; two calls produce a joint target without dropping a call; submit stays group 4; full-prefix+target exactly recovers original IDs; span merge preserves text and respects 24; tied multiple spans give 0.55; singleton gives 1 following pinned upstream; negative deltas remain ranked; missing action records `no_action_unscored`; ambiguous field boundaries fail.
- [ ] Run `.venv/bin/python -m pytest tests/test_span_lora_data.py -q`; verify tests fail for the missing new interface.
- [ ] Implement the interfaces and CLI CPU manifest export, reusing verified serializer/annotation semantics without changing historical files. Pin WebKD scorer semantics/revision in the new manifest.
- [ ] Run the same test file; require PASS. Export provenance and verify exact IDs/masks against all 1.180 corpus records with no GPU/tool execution.
- [ ] Commit only Task 1 files and its new manifest schema documentation.

### Task 2: Resumable offline scoring and reference cache

**Files:** Create `src/workflow_distillation/lora_fp16/span_scoring.py`, `src/workflow_distillation/lora_fp16/span_cache.py`, `scripts/training/score_span_lora.py`, `tests/test_span_lora_scoring.py`.

**Interfaces:** `score_turn(model: object, tokenizer: object, turn: dict) -> dict` returns full/ablated action logps, deltas and span weights. `cache_reference(model: object, input_ids: object, groups: object, chunk_size: int) -> object` returns float32 target logps in supervised-token order. `load_sidecar(path: Path, corpus_hash: str, reference_identity: dict) -> dict` validates complete files and mappings before training. Sidecars contain `ref_logp.bin`, `weights.bin`, `anchor_weights.bin` and a manifest with per-record offsets/checksums/status. Anchor weights are float32, independently mapped from rationale/envelope provenance.

- [ ] Write failing tests with a tiny deterministic model: analytic action-logp differences; fixed target sequence after removal; prefix-cache results match uncached results if optimization enabled; reference cache uses original prefixes; overlap mapping; interrupted write resumes exactly once; wrong model/template/hash/length or NaN/Inf fails; non-reasoning weights equal 1.
- [ ] Run `.venv/bin/python -m pytest tests/test_span_lora_scoring.py -q`; verify failure for missing interfaces.
- [ ] Implement scoring under eval/no-grad and atomic per-record completion. Reuse chunked output-head computation; never store full-vocabulary distributions. CLI exposes estimate/resume and explicit reference identity.
- [ ] Run the same tests; require PASS. Count eligible turns/spans on CPU and report estimated number of scoring variants, cache bytes and omitted/unscored turns.
- [ ] Commit only Task 2 files. Later, with a GPU available for this experiment, measure representative real-turn timings before committing to all-pool compute; report actual projected hours/cost rather than promising a budget.

### Task 3: Cached-anchor weighted token objective

**Files:** Create `src/workflow_distillation/lora_fp16/span_loss.py`, `tests/test_span_lora_loss.py`; modify `src/workflow_distillation/lora_fp16/config.py`, `engine.py`, `cli.py`, `loss.py` only at the new-objective integration points.

**Interfaces:** `chunked_span_loss(hidden: object, head: object, labels: object, groups: object, weights: object, ref_logp: object, anchor_weights: object, beta: float=0.03, chunk_size: int=128) -> tuple[object, dict]`. Returns differentiable trajectory loss plus detached sum/count metrics. New config mode `offline_span_anchor`; preflight validates compatible complete sidecar. Existing `vanilla`, `token_mean`, `token_group` behaviors remain unchanged.

- [ ] Write failing tests: dense analytic value and gradients match chunked loss; all weights 1/beta 0 reproduces vanilla; eval current=reference gives anchor 0 and gradient 0; rationale/envelope gamma equals 1 and other response gamma equals 1e-6; no-reasoning records retain envelope/epsilon support; empty anchor support returns 0; masks and shift are exact; near-zero/extreme d finite in FP32; unequal trajectory lengths retain trajectory mean and short final accumulation is correct.
- [ ] Run `.venv/bin/python -m pytest tests/test_span_lora_loss.py -q`; verify failure for missing new objective.
- [ ] Implement spec formulas with `expm1(d)-d`, FP32 reduction, independent anchor mask and CE weights. Consume q from cache; no second reference model in train. Aggregate/log weighted CE, plain CE, anchor, per-group NLL and effective weight mass.
- [ ] Run Task 1–3 test files plus existing trainer loss/accumulation tests identified by `rg --files tests`; require PASS for new and historical objective checks.
- [ ] Commit only new-objective code/tests. Do not rewrite hash-pinned old renderer/loss modules or old receipts.

### Task 4: Run profile, recovery/log export and evaluation handoff

**Files:** Create `configs/training/gpus/rtx5090_32gb/metacurate_span_anchor_bf16.json`, `scripts/training/run_metacurate_span_anchor.sh`, `tests/test_span_lora_run.py`; modify `src/workflow_distillation/lora_fp16/checkpoints.py`/`cli.py` only where necessary for the spec policy; document commands in `configs/training/gpus/rtx5090_32gb/METACURATE_SPAN_ANCHOR.md`.

**Interfaces:** Shell entrypoint checks corpus+sidecar identity, prints pinned run config, then calls the existing training CLI in the new mode. Metrics are exported as JSONL with update, epoch, objective components, per-group statistics, LR, throughput, peak VRAM and run identity. HF upload bundles include epoch adapter, config, metrics and manifest.

- [ ] Write failing tests: reject missing/partial sidecar before GPU training; preserve endpoint during pruning; upload retry never removes last local copy; epoch-only uploads; resume preserves update/scheduler/data order and appends logs without duplicated completed updates; no validation metric is fabricated for no-holdout training.
- [ ] Run `.venv/bin/python -m pytest tests/test_span_lora_run.py -q`; verify expected missing-policy failures.
- [ ] Implement only missing integration/policy pieces and the separate run profile. Reuse existing checkpoint mechanisms where they already satisfy these assertions.
- [ ] Run all four new test files and relevant existing checkpoint/resume tests; require PASS. Check shell entrypoint with `bash -n scripts/training/run_metacurate_span_anchor.sh` and config loading on CPU.
- [ ] Commit only Task 4 files. Publish code only if separately requested; preserve unrelated working-tree changes.
- [ ] At execution time after scoring, run one longest-trajectory forward/backward memory check, then enter epoch without extra repeated pilots. Keep two epoch endpoints; report current metrics/cost and compare against vanilla using identical inference settings. CE-only is an optional ablation, not a prerequisite that silently doubles spend.

## Self-review and handoff

Implementation completed and CPU-verified on 2026-10-27; GPU admission remains a server-time step.

Spec coverage checked: scoring, exact alignment/omission, reference cache, formulas,
reduction, memory, cost estimate, logs, checkpoints and comparison all have owning
tasks. Review Focus conditions are covered by the named task tests. At initial plan authoring, no training, API call, implementation, commit or push was performed.
