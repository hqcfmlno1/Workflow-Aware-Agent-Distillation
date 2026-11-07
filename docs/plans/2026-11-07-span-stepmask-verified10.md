# Span-anchor + step-mask: prepare Verified10

Prepare only, with no GPU inference or modification of the running training job.
Reuse the exact original ten frozen Verified rows, excluding the masked training
pool. Only the new span-stepmask epoch-2 checkpoint (step 590) is eligible.

- [x] Stage clean local runner, frozen tasks and isolated inference bundle.
- [x] Finalize explicitly: pin HF commit, verify masked training identity and
      export only adapter weights; missing checkpoint must stop, never fallback.
- [x] Preserve step resume, 200 actual tool executions, 1200 s/generation,
      1800 s/task, sampling 0.6/0.95/20, BF16 and native context.
- [x] Test preparation, source validation, no gold data in server, and reports.

Docker tasks/official grading remain local; inference uses a separate remote
venv after training finishes. No campaign budget. Task-local failures continue;
Ctrl-C preserves the unfinished step. Old histories/outcomes are never imported.

Implementation uses the operational 200-tool runtime snapshot, records its source
hashes and packs a self-contained archive. No live model calls during preparation.
No delegated review: user requested conserving quota; review locally.

Completed locally: six focused new checks and eight existing Verified10 checks
passed (14 total); Ruff and shell syntax checks passed. Offline stage created the
ten-task bundle with `adapters_ready=false`. Finalization code is prepared but
has not contacted HF or fetched checkpoint 590 during this request.

Self-review: checked source/manifest pinning, masked index + sidecar provenance,
isolated BF16 export labels, request model identity, task timeout/step resume,
official grading versus partial workspace diffs, owned Docker cleanup and the
deployment gate against concurrent GPU training. No GPU inference was started.

Publication preparation: runtime snapshot is now tracked under
`cloud/vast/verified10_stepmask/runtime/` with original source hashes. A clone
can fetch the pinned Verified rows without historical output folders; masked
corpus and sidecar manifests remain separately restored experiment inputs.
Final focused verification before commits: 51 tests passed, benchmark source
Ruff clean, shell syntax clean, and offline stage succeeded without model calls.
