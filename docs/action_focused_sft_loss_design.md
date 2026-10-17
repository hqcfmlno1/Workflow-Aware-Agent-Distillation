# Action-focused SFT — Phase 1 loss design

**Qwen3.5-4B · mini-SWE-agent · 4,000-trajectory baseline**

Updated: 2026-10-15 · Design version: 2.0

## 1. Objective and scope

Phase 1 adapts an already capable agent through conditional imitation of teacher reasoning, commands, and terminal submission. Tool-call supervision teaches syntax, command and argument selection, and their use given the preceding teacher history. A useful published SWE-bench score motivates this broader goal; it does not establish a syntax-validity rate for our exact configuration.

The baseline uses **SmartAD-style weighted-token cross-entropy** with hard teacher targets. Starting reasoning/content/action/submit weights are **1.0 / 0.75 / 1.5 / 2.0**. Reasoning remains in both input and targets wherever the official template renders it. System messages, user messages, observations, padding, and supplied assistant prefixes are context only.

These weights are initial hypotheses, not empirically optimal SWE values. No teacher logits are required. The objective applies to LoRA or QLoRA, while numerical stability and memory require remote profiling. The local machine prepares data and stores assets; training runs remotely.

## 2. Relation to SmartAD

SmartAD Section 3.3 uses reason/action/final token weights **1 / 1.5 / 2** and normalizes the weighted CE sum by the sum of token weights. This design adopts that token weighting and normalization, adds content with weight 0.75, and maps final to the terminal submission call. The equal-trajectory reduction below is an explicit baseline choice. Report these adaptations rather than claiming an exact replication.

Within a trajectory, an action token receives 1.5 times the coefficient of a reasoning token; a submit token receives twice that coefficient. Content has a modestly lower coefficient. Group contributions depend on token counts and CE values. Longer reasoning can contribute more to the objective. Coefficients do not specify fixed group percentages, gradient shares, memory use, or runtime.

Reasoning targets also discourage switching to tools where the teacher continues reasoning. This supervision remains substantial. It is an implicit teacher-forced signal, not a direct penalty on generated rollouts or a guarantee of correct timing. Shorter reasoning than the teacher is not necessarily wrong. Nonzero reasoning loss does not guarantee preserved thinking capabilities.

Rule-based selection is a quality/complexity proxy. It does not implement SmartAD's student-NLL selection or establish capability alignment.

## 3. Token groups and masks

Every supervised assistant target belongs to exactly one group.

| Group | Target region | Weight per token |
| --- | --- | ---: |
| R — Reasoning | Assistant reasoning actually rendered by the template | 1.0 |
| C — Content | Assistant prose outside reasoning/tool calls, including summaries | 0.75 |
| A — Action | Ordinary tool calls and model-generated action protocol delimiters | 1.5 |
| F — Terminal submission | The final tool call that terminates and submits the trajectory | 2.0 |
| Context/padding | System, user, tool schemas/observations, padding, supplied assistant prefixes | Masked: `labels = -100` |

A and F include the complete tool-call wrapper, function name, arguments, command, and code written through the command. Code or patches in observations remain context. Reference patches must not be added as input or new targets. Reasoning and prose in the final assistant message retain R and C weights.

Partitioning rules:

1. Use source roles, structured fields, and verified render positions. Do not assign labels by scanning all text for submit strings or code fences: these also occur in prompts, reasoning, and observations.
2. Assign F only to the actual terminal submission call. Earlier failed submit attempts remain A. Classify each call separately in multi-call messages.
3. Permit terminal submission without a final observation, as in the audited data. Do not fabricate observations or patch targets.
4. Preserve multiline commands, Unicode, quotes, and escapes. Decode the intended JSON layers once; do not rewrite commands to simplify training.
5. Mask assistant prefixes already supplied at inference. Model-generated phase-transition delimiters such as `</think>` belong to A. Tool-call delimiters inherit A or F from their call. The message-ending delimiter after submission is F; after ordinary actions it is A. For messages without calls, reasoning delimiters inherit R and content-only message endings inherit C. Version the exact mapping against the inference prefix and official template.
6. Whitespace follows its rendered region. Reject ambiguous or unassigned assistant target boundaries rather than silently assigning an action label.

Equal coefficients do not imply equal loss or gradient contributions. Already predictable syntax can contribute less signal than uncertain command arguments. Long edit payloads still contain many targets and may dominate other actions. The objective does not equalize action steps or distinguish useful from redundant actions inside successful trajectories.

## 4. Terminal submission and patch protocol

The initial prompt in mini-SWE-agent v2.4.6's SWE-bench configuration asks the agent to create and inspect `patch.txt`, then submit with:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

The Docker environment detects the marker on the first output line when the command return code is zero and takes remaining output as the submission. The student chooses when to submit. Protocol acceptance does not establish patch correctness.

Ordinary `mini.yaml` instead requests the marker-only command without another command. SWE-bench Verified requires a patch prediction for evaluation, but does not universally require a file named `patch.txt`; collection depends on the harness configuration.

Retain `cat patch.txt` in this baseline's terminal targets with weight 2.0, using the corresponding patch-output protocol for evaluation. This can teach a protocol-specific habit; it does not establish generalization to prompts without patch-file requirements. Masking only `cat` would also leave earlier patch-creation actions supervised.

The 2026-10-13 audit of 49,925 successful trajectories found submit instructions in every initial prompt and one final Bash submission call per trajectory. Its 23 command variants comprise 35,586 exact commands, 11,828 with a `cd /testbed &&` prefix, 2,473 using `/testbed/patch.txt`, and 38 others. Preserve these commands. SmartAD's final block and this terminal handoff have different semantics; weight 2.0 remains a starting experiment.

## 5. Official template and token boundaries

Use `Qwen/Qwen3.5-4B` tokenizer revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with chat-template SHA256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`. Existing records use `add_generation_prompt=False`.

The pinned template renders assistant reasoning after the last user query; reasoning before a newer user query can be omitted. Do not assume only the final response has reasoning or every normalized `reasoning_content` field has rendered tokens. Report omitted reasoning. Exposing it through changed sample construction requires a versioned representation.

Tokenize complete rendered text with the official tokenizer. Convert verified character spans using offset mappings and explicit special-token handling. Separately tokenizing and concatenating regions can change token IDs. Resolve ambiguous cross-region tokens and unmatched offsets before training. Existing `action_candidates` are character spans, not completed token labels.

## 6. Loss formula and reduction

For target token i in trajectory n, teacher-forced CE is:

$$\ell_{n,i}(\theta)=-\log p_\theta(y_{n,i}\mid h_{n,<i}).$$

Let I(n,g) contain supervised positions in group g, with count N(n,g). Its CE sum is:

$$S_{n,g}=\sum_{i\in I(n,g)}\ell_{n,i}.$$

The weighted target count and trajectory loss are:

$$D_n=N_{n,R}+0.75N_{n,C}+1.5N_{n,A}+2N_{n,F},$$

$$L_n=\frac{S_{n,R}+0.75S_{n,C}+1.5S_{n,A}+2S_{n,F}}{D_n}.$$

For B trajectories, retain equal trajectory weighting:

$$L_{\mathrm{batch}}=\frac{1}{B}\sum_{n=1}^{B}L_n.$$

Accumulate CE sums, multiply by token weights, and normalize by weighted target counts within each trajectory. Empty groups contribute zero to numerator and denominator. Reject trajectories without supervised targets. Missing A or F is a data error for this corpus; absent R must be checked against source reasoning and template behavior.

Scaling all four coefficients by the same positive constant leaves loss unchanged. Changing one coefficient changes its contribution and the normalizer. Weighted totals from different coefficient configurations are not a common model-quality metric.

### 6.1. Windows, accumulation, and DDP

The formula defines the objective over complete represented trajectories. If trajectory n is split into chunks s, supervise each target once and mask repeated history. Keep the full-trajectory denominator D(n):

$$q_{n,s}=\frac{\sum_{i\in I(n,s)}w_{n,i}\ell_{n,i}}{D_n},\qquad L_n=\sum_s q_{n,s}.$$

Independently normalizing chunks and averaging changes the objective. For N trajectories and K chunks, uniform chunk sampling with sample loss (K/N) times q(n,s) is an unbiased estimator of the trajectory mean. Other sampling distributions require corresponding probability weights. Equivalence assumes identical conditioning context. Shorter context or dropped targets changes the training problem and must be documented.

The initial remote run will retain complete trajectories that fit a measured training context limit and exclude longer trajectories from that run. Keep excluded originals for later work; do not truncate histories, silently replace excluded tasks, or enable sliding windows. The context threshold and resulting training count must be recorded before baseline training. Splitting a trajectory into next-step examples with full preceding history does not shorten late-step inputs: the input prefix still grows with each step. Such examples therefore need their own context policy.

Implementation requirements:

- Use per-token CE with `reduction='none'`, `ignore_index=-100`. Causally shift predictions, labels, groups, and weights exactly once.
- Reduce weighted CE and denominators in FP32. This does not fix forward-pass instability.
- Separate attention and loss masks: observations remain visible as context.
- Preserve reduction across DDP ranks, uneven final batches, and gradient accumulation. Averaging unequal local means is insufficient.
- Packing must preserve trajectory boundaries, prevent cross-task attention, and retain each trajectory's denominator. Backend support is unverified.

## 7. Training scope and checkpointing

The rule-selected source baseline contains **4,000 trajectories**. The requested training endpoint is **one epoch** over the explicitly finalized context-fitting subset, with fixed loss weights. Filtering changes the effective training count; report that count separately from the 4,000 source trajectories. General-data replay is not part of this initial baseline. One epoch does not establish that forgetting is absent or that training fits the compute budget.

Save full training checkpoints: adapter, optimizer, scheduler, global step, and data/RNG state. Record consumed examples to prevent accidental replay on resume. Loading adapter weights alone starts a new optimization stage. Any future continuation would require an explicit data and learning-rate schedule; no dataset expansion or second epoch is currently planned.

The optimized QLoRA and one-batch DDP pilots implement this policy with per-rank optimizer/GradScaler/RNG state, an explicit next-data cursor, atomic checksummed saves, an interval of five completed updates, two retained local copies, and optional private Hub backup. CPU single-process and two-rank Gloo tests cover exact continuation; CUDA/bitsandbytes resume remains unverified. The DDP pilot still performs just one total update and forces its final save. The [current configuration](qlora_current_config.md) uses r=32/alpha=64 and PagedAdamW32bit; earlier GPU results used r=16/alpha=32 and PagedAdamW8bit. See the [checkpoint/resume guide](training_checkpoint_resume.md) for configuration, recovery, and session-loss limitations. This infrastructure does not constitute the complete baseline trainer.

## 8. Evaluation and checkpoint policy

For the first fixed one-epoch experiment, evaluate the **final checkpoint** on all 500 SWE-bench Verified tasks. Teacher-trajectory validation loss is optional diagnostic information, not the checkpoint-selection criterion. Retain the existing 50 held-out NVIDIA tasks and exclude all their attempts from training, including expansions.

Freeze data, weights, training endpoint, and evaluation settings before running. Compare base and adapter with the same harness version, patch protocol, prompt, generation settings, environment, step/token budgets, and task set. Published scores are context, not a controlled comparison. Report resolved rate, resource use, and protocol/infrastructure failures.

If diagnostic teacher-forced CE is computed, report unweighted R/C/A/F CE with a fixed aggregation convention, token counts, and weighted total loss. These measure imitation on teacher states. They do not establish rollout success; matching submit syntax does not establish patch correctness.

Repeatedly selecting weights, checkpoints, or extensions using Verified results turns it into development data. Describe subsequent results accordingly or use a separate untouched test set. Reserve remote budget for base/adapter inference and benchmark execution. Runtime within the 30-hour Kaggle quota is unverified.

## 9. Checks required before training

1. Every supervised assistant token has one group and its weight 1.0, 0.75, 1.5, or 2.0. Context/padding have `labels=-100`; A/F do not overlap.
2. Official token IDs and tool payloads are preserved, including multi-call messages, Unicode, multiline commands, quotes, and special-token boundaries.
3. Submit strings in prompts, reasoning, prose, or observations do not receive F. Earlier failed submits remain A; terminal calls without observations work.
4. Numerical checks cover weighted CE sums/counts, empty groups, masks, causal shifts, and invariance to coefficient scaling. Duplicated reasoning targets increase their weighted count as specified by the formula.
5. Chunk contributions reconstruct full-trajectory loss under identical context. Sampling, accumulation, and one/two-GPU reduction preserve the objective.
6. A remote backward smoke test produces finite gradients and updates adapters. Resume preserves optimizer/scheduler state and data position.

The remote numerical pilot checks weighted CE and gradients against a full-logit reference. Later checks cover cross-device vocabulary projection, checkpoint dropout RNG on two GPUs, and internally blocked Gated DeltaNet output/state/input gradients under nested checkpointing and CPU offload, including GPU FP16 autocast. Optimized QLoRA completed two finite updates at 16,320 tokens on both one T4 and a partitioned pair, before OOM at 32,766 tokens in the unblocked implementation. Internal compute blocking preserves full histories and state gradients; the [Kaggle instructions](../cloud/kaggle/README.md) record subsequent results. These are feasibility probes, not full-state resume checks. Baseline SFT and benchmark evaluation have not started.

## 10. Data status and provenance

The baseline contains 4,000 trajectories from distinct tasks that originally each had at least two successful attempts. Files are under `data/processed/mini_swe_agent_rule_baseline_4000/`, containing normalized messages, rendered text, token counts, and action candidates. CPU token-group preparation is complete under `data/processed/weighted_sft_tokens_4000/`: all 4,000 renders, token counts, target counts, and binary checksums were verified, with no overlap with the 50 held-out tasks. A remote chunked weighted-loss implementation and full-history QLoRA memory probes are available. The effective training count, complete baseline trainer, CUDA full-state resume verification, and full-corpus runtime budget are not finalized. See [Kaggle instructions](../cloud/kaggle/README.md) for commands and measured results.

The baseline has 191,891,858 serialized tokens and a median of 40,715 tokens per trajectory. These are input-text counts, not supervised-target counts or measured runtime. Reducing loss weights does not remove context computation. Success-only SFT does not guarantee recovery from student errors or improvement over base. Exact teacher-token CE gives no explicit credit for alternative valid commands, and successful teachers can contain mistakes.

Sources and pinned artifacts:

1. [SmartAD, Section 3.3 — Segment-Weighted Agent Loss](https://aclanthology.org/2026.findings-acl.1349.pdf).
2. [mini-SWE-agent v2.4.6](https://github.com/SWE-agent/mini-swe-agent/releases/tag/v2.4.6).
3. [SWE-bench prompt configuration](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/config/benchmarks/swebench.yaml).
4. [Ordinary mini configuration](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/config/mini.yaml).
5. [Prompt construction](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/agents/default.py) and [Docker submission detection](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/environments/docker.py).
6. [NVIDIA Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces), v1.2, split `minisweagent`, revision `e192b26a43b8d09f156eeee781aef8e6bd5108ba`; Qwen3.8-27B Thinking, `resolved=1`.
7. [Pinned Qwen3.5-4B tokenizer](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a).
8. Local submit audit: `data/interim/nvidia_success_submission_audit_2026-10-13.json`.
9. Baseline manifest: `data/processed/mini_swe_agent_rule_baseline_4000/task_selection_manifest.json`; SHA256 `dba8102cbd90be706a48dc244b444718998f974228077fac0a5a7ee564e628fe`.

Trajectory data stays local and is not embedded in this document or committed.
