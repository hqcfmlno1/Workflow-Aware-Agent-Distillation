# DeepSeek 1,180 — LLM whole-step loss masking

Policy: [step_mask_policy_v1.txt](../../configs/data/step_mask_policy_v1.txt).
Runner: [judge_trajectory_steps.py](../../scripts/data/judge_trajectory_steps.py).

One complete trajectory per API request; no source truncation, no command
execution, no rewrites, no training. Scope is the frozen 1,180 successful
DeepSeek V4 Pro trajectories with student-rendered length at most 36k tokens.
The corpus totals 24,256,659 student tokens; this is **not** an API-token bill.

The approved policy masks whole assistant steps, not individual reasoning spans.
One step is one assistant message, potentially containing multiple tool calls.
Step IDs start at 1. Original message indices start at 0. Observations remain
context and are never targets of these mask decisions. Unlisted assistant steps
keep their existing supervision only for successfully judged trajectories.

The judge returns only `trajectory_id` and `mask_steps`. Each mask includes
`step_id`, `reason_codes`, `reason`, and literal `source_id`/`quote` evidence.
The validator requires valid unique IDs, permitted reason codes, exact evidence
matches, and at least one quote from the masked assistant message itself.
This structural validation does not independently prove the judge is correct.

## Run and monitor

API key is read from a private file outside Git:
`~/.cache/workflow-distillation/llm-step-mask.key`. Never put it in commands,
logs, prompt text or exported annotations.

From the repository root:

```bash
# Prepare/check resume state without calling the API.
.venv/bin/python scripts/data/judge_trajectory_steps.py

# Run or resume. The key must have permission to use the requested model.
.venv/bin/python -u scripts/data/judge_trajectory_steps.py --run

# Follow progress in a second terminal.
tail -f outputs/data_audits/deepseek1180_llm_step_mask_v1/campaign.log

# Read compact persistent progress.
cat outputs/data_audits/deepseek1180_llm_step_mask_v1/summary.json
```

Default endpoint is `http://localhost:20128/v1`; model discovered from its
catalog is `cx/gpt-6-luna[1m]`. Uses Responses API, medium reasoning, 12k output
token budget and two concurrent requests. No tools are supplied to the judge.
Flags can change transport, output, retry and concurrency settings. A changed
model/policy/request configuration requires a different output directory, so
previous decisions cannot be silently mixed with new decisions.

`Ctrl-C` or `kill -TERM <pid>` stops dispatching and lets in-flight requests
finish. Restart the same command to resume completed records without recalling
the API. An interrupted in-flight request without a completed local receipt
may need to be called again; the proxy provides no guaranteed job recovery.

## Files

All artifacts live under `outputs/data_audits/deepseek1180_llm_step_mask_v1/`:

- `manifest.json`: frozen corpus/policy hashes and request settings.
- `campaign.log`: trajectory start/finish, retries, and 15-second heartbeats.
- `summary.json`: completed/pending/errors, active trajectory IDs and durations,
  total masked steps and returned usage from successful receipts.
- `results/<id>.json`: atomic validated annotation and source/request hashes.
- `attempts/<id>/*.json`: attempt status, successful raw API responses, returned
  usage even for invalid annotations, and validation errors.
- `mask_steps.jsonl`: completed sparse annotations plus original message indices.
- `annotated_trajectories.jsonl.gz`: original source record and separate loss-mask
  metadata; original messages remain unchanged. This is **not** tokenized data.

Exports refresh every 20 completions and on clean termination. Per-trajectory
receipts and `summary.json` update immediately, so exports can lag live progress.
An API/validation error is never converted to an empty mask. Those trajectories
stay pending. HTTP 401/403/404 stops new dispatch rather than failing the entire
pool repeatedly. On resume the runner retries records without valid receipts.

Applying these annotations to token labels is a separate preparation step. It
must respect the original tokenizer/template mappings and mask only assistant
tokens at the recorded message indices. No existing training corpus or trainer
has been changed by this run.

## API permission at initial setup

The model catalog succeeded, but the first trajectory request returned HTTP 403:
`permission_error / insufficient_quota`, with the message that the key was not
allowed to use `cx/gpt-6-luna`. No trajectory judgment completed at that point.

After the user changed proxy permissions, `cx/gpt-6-luna[1m]` accepted the
first trajectory: 18,713 reported input tokens, 25 output tokens, validated empty
mask list, 2.318 seconds. The complete run was started with two workers, reusing
that completed receipt. The proxy response does not echo a model ID or reasoning
settings; the configured alias and requested effort are recorded in receipts.
