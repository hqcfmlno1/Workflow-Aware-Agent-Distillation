# Offline CoReD for agent trajectories

This implements the agreed **WebKD + harmonic key-span selection + CoReD**
adaptation. It is not an exact reproduction of the paper's long-CoT recipe:
it keeps complete agent histories, protects action supervision, and uses the
existing action-informativeness span weights in both training phases.

## Data and objectives

Input is an immutable token/group corpus, its exact character-offset provenance,
and a complete WebKD scoring directory **including original `parts/*.turn-*.json`**.
The converter checks corpus/provenance/sidecar hashes, target order, original span
weights and token coverage. It does not apply a chat template again. Truncated
and step-masked corpora are rejected. Existing omissions made by the original
chat template are recorded in trajectory metadata; conversion adds no omissions.
System/user/observations remain context;
all originally supervised assistant action, content, final and ending tokens
remain targets. A token crossing two character spans has fractional membership,
whose total supervised mass is exactly one. Unassigned assistant protocol tokens
receive response weight one.

Let `s` denote a reasoning span, `v[s]` its existing WebKD weight in `[0.1,1]`,
and `H[s]` the frozen backbone's mean full-vocabulary entropy on its target
positions under the original teacher-forced history. Within **each trajectory**:

```text
rank(x)[s] = (average_ascending_rank(x[s]) - 1) / (number_of_spans - 1)
a[s] = rank(v)[s]
e[s] = rank(H)[s]
J[s] = 2*a[s]*e[s] / (a[s]+e[s])
K = largest ceil(0.1*number_of_reasoning_spans) values of J
```

Singleton rank is one; zero denominator gives zero. Ties use average rank, then
source span order for the cutoff. Higher `v` means greater action relevance in
the existing scoring implementation. No extra relevance threshold is introduced.
Actions are excluded from selection. These selections are fixed before training.

### Phase 1: council

A shared unquantized frozen backbone carries independent FP32 LoRAs. Default
council size is three, rank 32, alpha 64. Model/LoRA stochastic dropout is disabled;
reasoning **supervision** dropout independently retains each span with probability
0.9, keeping all its tokens in context. Response/action targets are never dropped.
For each expert, NLL is the weighted token mean in each trajectory, then mean over
trajectories. Expert losses are **summed**, not averaged across experts.

```text
L1 = sum_m(mean_trajectory(weighted_token_mean(NLL_m)))
     + lambda_rep * mean_trajectory(Lrep)
     + lambda_ortho * Lortho
D[m,n] = sum_selected_tokens(sum_vocab((log p_m - stopgrad(log p_n))^2))
Lrep = sum_m(sum_n!=m(exp(-gamma*D[m,n]))) / (M-1)
```

`Lortho` sums ordered expert pairs and adapted layers. For factors `A_m,B_m`:

```text
ortho(m,n) = ||A_m*A_n^T||_F^2 / (||A_m||_F^2 * ||A_n||_F^2)
             + ||B_m^T*B_n||_F^2 / (||B_m||_F^2 * ||B_n||_F^2)
```

Each squared norm is clamped below at `1e-8`. Defaults: `lambda_rep=1`,
`lambda_ortho=0.01`, `gamma=1`. Repulsion uses full vocabulary and a **sum** over
tokens; it can saturate for large distances. This is not target-token k3 anchoring.
Detached peer predictions are captured before any expert update. Each expert's
forward/backward finishes before switching adapters, including gradient-checkpoint
recomputation. All experts share one optimizer step; orthogonality is added once
per accumulated update.

### Frozen council and TIES

For every reasoning span in the training corpus:

```text
U[s] = mean_tokens(mean_experts(H(p_m)))
V[s] = mean_tokens(H(mean_experts(p_m)) - mean_experts(H(p_m)))
rho[s] = V[s] / (U[s]+epsilon)
z(x[s]) = (x[s]-population_mean(x)) / (population_std(x)+epsilon_std)
wC[s] = 1 + 0.5*tanh(z(U[s])) + 0.5*tanh(z(rho[s]))
wH[s] = v[s]*wC[s]
```

Moments are fitted across **all training reasoning spans**, not per trajectory.
Full-vocabulary probabilities define entropy and Jensen-Shannon disagreement;
there is no sampled-vocabulary approximation.

TIES trims each expert's effective update `(alpha/r)*B@A` per adapted layer,
elects the sign by the sum of retained updates, and averages only retained
nonzero updates with that sign. A zero elected sign yields zero. The density
cutoff retains exactly `ceil(density*layer_numel)` entries, with stable index ties.
Default density is 0.2 and merge scale is 1. The standalone **post-TIES model**
is saved separately and hash-verified. Arbitrary elementwise TIES can increase
rank, so the implementation folds the exact dense merged update into the base;
it does not silently approximate it with rank-32 SVD. FP32 merge results are
rounded to the base dtype on storage.

### Phase 2: one student

The post-TIES model becomes the frozen base/reference; a fresh zero-update LoRA
is the student. Initial KL is therefore zero. The council stays fixed offline.
For each trajectory, NLL and KL each use a weighted mean of **per-segment token
means**. Reasoning weight is `wH`; response/action/protocol segment weight is one.
Then average across trajectories:

```text
L2 = mean_trajectory(sum_segments(wH[s]*(mean_tokens(NLL[s])
                        + lambda_KL*mean_tokens(KL(q_ref || p_student))))
                     / sum_segments(wH[s]))
```

For response segments substitute weight one. `KL(q_ref || p_student)` is the exact
full-vocabulary forward KL, with the merged reference detached; default coefficient
is one. This differs from the old span-anchor target-token k3 loss. Both vocabulary
projection losses use checkpointed token chunks to avoid keeping a sequence-length
by vocabulary tensor in memory. Decoder activations and dense TIES still consume
memory; no 32GB feasibility or runtime estimate is claimed before GPU measurement.

## Commands

Install the project's training dependencies. Use local model snapshots containing
weights and tokenizer files. No command downloads models or starts remote jobs.
All outputs must be inside an ignored `draft/` directory. Configuration overrides
are an optional JSON object matching `CoRedConfig`, also stored in `draft/`.
Omitting `--config` uses its explicit defaults. Use the same recipe file through
Phase 1, scoring and merge; Phase 2 may have its own training recipe.

```bash
python -m workflow_distillation.cored prepare \
  --corpus-index /path/original/index.json \
  --provenance-index /path/provenance/index.json \
  --webkd /path/full-webkd-scoring-store --output draft/cored/data

python -m workflow_distillation.cored score-base \
  --model /path/base --data draft/cored/data --output draft/cored/selected

python -m workflow_distillation.cored phase1 \
  --model /path/base --data draft/cored/selected --output draft/cored/phase1

python -m workflow_distillation.cored score-council \
  --model /path/base --data draft/cored/selected --output draft/cored/weighted \
  --checkpoint draft/cored/phase1/checkpoint-XXXXXXXX

python -m workflow_distillation.cored merge \
  --model /path/base --data draft/cored/selected --output draft/cored/merge \
  --checkpoint draft/cored/phase1/checkpoint-XXXXXXXX

python -m workflow_distillation.cored phase2 \
  --model draft/cored/merge/post-ties --data draft/cored/weighted \
  --output draft/cored/phase2
```

Training defaults: trajectory microbatch one, accumulation four, two epochs,
AdamW LR `5e-5`, 10% linear warmup then linear decay, gradient clipping one,
checkpoint every ten optimizer steps and at completion. BF16 base/FP32 adapters;
`dtype=float32, --device cpu` supports small local verification. Qwen3.5 supports
explicit `gdn_backend=reference` or `fla`; there is no silent FLA fallback.

Use `--resume /same/output/checkpoint-XXXXXXXX` on `phase1` or `phase2` to restore
adapters, optimizer, schedule cursor, reasoning-dropout RNG and Torch/CUDA RNG.
Checkpoints are committed only after successful updates; a killed partial update
replays from the last complete checkpoint. Scoring resumes from verified per-record
parts. Changed data/model/config identities and corrupted artifacts are rejected.
One filesystem lock protects each stage against duplicate writers.

Each stage writes `events.jsonl`; training also writes `completed.json`,
`latest.json`, recovery checkpoints and adapter exports. For deployment use the
post-TIES base plus the final `phase2/adapters/expert0` PEFT adapter. Phase 1 council
adapters are under `phase1/adapters/expert{0,1,2}`. Keep both post-TIES and final
student artifacts; the final LoRA is relative to the post-TIES base, not the
original backbone. There is no automatic upload, benchmark or instance shutdown.

Local checks and diagnostic logs for this implementation live under ignored
`draft/cored-implementation-20261116/`; server validation is intentionally deferred.
