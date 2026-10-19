# Dữ liệu cục bộ

Máy local chỉ chuẩn bị dữ liệu/tokenizer trên CPU. Fine-tune chạy ở môi trường khác.
Xem [pipeline và metric phase 1](../docs/action_focused_sft_phase1.md).
Phase 1 mang tên **Action-focused SFT**; xem [thiết kế loss](../docs/action_focused_sft_loss_design.md)
và [notebook phân tích trajectories](../notebooks/nvidia_trajectory_action_focused_sft.ipynb).

## Đầu vào cho first run hiện tại

| Thư mục | Vai trò |
| --- | --- |
| `processed/first_run_epoch_sft/` | `index.json`, `tokens.bin`, `groups.bin`: nguồn token/group gốc, 11.527 trajectory; giữ nguyên bytes/hash để resume |
| `processed/first_run_diverse_4000/` | Manifest cohort đến batch 2.000, giữ prefix tại 695; 4.000 trajectory / 3.866 task / 1.501 repo; tham chiếu binary gốc |
| `processed/nvidia_mini_swe_agent_v1_2_success_qwen35_4b/` | Corpus rendered nguồn để tái tạo token dataset |
| `interim/nvidia_mini_swe_agent_v1_2_success/` | Successful exports trước normalization |
| `raw/qwen35_4b_tokenizer/` | Tokenizer/template pin theo model revision |
| `raw/omnicode_keys_source/` | Metadata benchmark để chống trùng; không phải trajectory OpenAI |

`first_run_diverse_4000/` gồm `diversity-plan.json`, `trajectories.jsonl`,
`summary.json` và `receipt.json`. Manifest đã xác minh binary nhưng chưa bind
checkpoint thật hoặc được tích hợp vào launcher Vast. Xem
[cách tạo và điều kiện resume](../docs/vast_diversity_continuation.md).

Luồng hiện tại:
`NVIDIA raw → successful export → Qwen render → token/group binary → cohort manifest`.
Lịch continuation Kaggle gốc ở `cloud/kaggle/recovery/continuation-plan.json`
được giữ nguyên để đối chiếu prefix và checkpoint identity.

Log, metrics và checkpoint nằm trong `outputs/training/first run/`.
Không lưu đầu vào train đã chốt trong `outputs/`; không sao chép binary vào
manifest cohort. Collector OmniCode/OpenAI cũ đã được gỡ; thư mục
`teacher_dataset/` không còn tồn tại trong workspace khi dọn ngày 2026-10-19.

## Các bộ dữ liệu và bước chuẩn bị trước đây

Các phần dưới mô tả pipeline nguồn và các tập pilot/selection trước đây;
chúng không thay thế cohort tiếp tục hiện tại.

| Thư mục | Nội dung |
| --- | --- |
| `raw/nvidia_open_swe_traces/` | Cache cột nguồn NVIDIA theo revision; không đọc reference patch |
| `raw/qwen35_4b_tokenizer/` | Tokenizer/chat template chính thức; không tải model weights |
| `interim/nvidia_mini_swe_agent_v1_2_success/` | Toàn bộ success trajectory của Qwen3.8-27B Thinking, raw Parquet + provenance/partition |
| `interim/` | Kết quả lọc và chuẩn hóa trung gian |
| `processed/mini_swe_agent_v1_2_phase1/` | 1.000 train + 50 validation; test ngoài là 500 task SWE-bench Verified, text, pool, checks và manifest |
| `processed/mini_swe_agent_rule_baseline_4000/` | Selection theo rule trước đây, khác cohort diversity tiếp tục hiện tại |
| `processed/weighted_sft_tokens_4000/` | Token/group cho selection theo rule trước đây, phục vụ pilot |
| `splits/` | Dành cho các split khác sau này; phase 1 lưu split cùng processed artifacts |
| `manifests/` | Dành cho các lần chuẩn bị khác; manifest phase 1 nằm cùng processed artifacts |

Bắt đầu từ `configs/datasets/nvidia_mini_swe_agent_v1_2_action_focused_sft.json`. Loại task trùng OmniCode trước khi chia train/validation; tất cả trajectory cùng repository + PR đã chuẩn hóa phải cùng split, kể cả khi instance_id khác định dạng giữa hai nguồn. Không đưa reference_patch vào đầu vào student.

Dữ liệu phát sinh không được commit. Các thư mục rỗng có thể không xuất hiện sau clone; bước chuẩn bị dữ liệu cần tạo chúng. Với Kaggle, gắn dataset lớn qua Kaggle Dataset dưới `/kaggle/input/`; cấu hình training phải nhận đường dẫn đầu vào thay vì giả định đường dẫn máy local.

## Toàn bộ success trajectories

```bash
uv run python -m scripts.data.export_success_trajectories --offline
```

Lọc `resolved == 1`, Qwen3.8-27B Thinking trên 44 shard mini-SWE v1.2 đã pin.
Giữ mọi category và attempts, messages/reasoning/content/tool calls/observations như
nguồn; không đọc reference patch. Nếu thiếu cache, bỏ `--offline` để đọc các cột cần
thiết từ Hugging Face (Parquet có thể truyền cả row group chứa các hàng không pass).
Mỗi part có checksum/receipt để tiếp tục khi gián đoạn; payload được kiểm tra round-trip.

`partition=train_pool` là ứng viên để chuẩn bị train sau này. Mọi attempts của task
validation hoặc trùng OmniCode vẫn có trong bản export nhưng được đánh dấu
`validation`, `excluded_omnicode` và phải loại khỏi train. File `report.json`
ghi số lượng và checksum. Đây là dữ liệu raw, chưa normalize/serialize theo Qwen hay
tạo labels; bộ processed giữ 1.000 train + 50 validation. Không chạy model hoặc command teacher.

Test cuối dùng 500 task `princeton-nlp/SWE-bench_Verified` theo revision trong config,
không dùng để chọn checkpoint/hyperparameter. Audit exact task overlap nằm ở
`reports/swebench_verified_overlap_2026-10-12/`. Bản test nội bộ 50 task và success
export trước đây được lưu lại trong `interim/archive_internal_test_2026-10-12/`;
đây là dữ liệu lịch sử, không đọc làm split hiện hành.

### Full success corpus rendered for Qwen3.5-4B

`data/processed/nvidia_mini_swe_agent_v1_2_success_qwen35_4b/` contains
`part-*.jsonl.gz`, one output record for every exported successful trajectory.
Run `.venv/bin/python -m scripts.data.convert_success_trajectories --workers 3`
to build/resume, and add `--verify` to replay every source record through the
pinned official tokenizer and compare the complete output.

Each valid record retains metadata, `partition`, normalized `messages` (including
teacher reasoning), parsed `tools`, validation `checks`, rendered `text`,
`token_count`, and character spans in `action_candidates`. No reference patches,
context truncation, packing, model inference, labels, or loss masks are added.
Historical schema-invalid calls remain recorded as invalid checks. Records that
cannot be normalized/rendered have `serialization_status="error"`, `text=null`,
original `raw_messages`/`raw_tools`, and an explicit error; consult `errors.json`.
Only valid records with `partition="train_pool"` are eligible for training.
All validation attempts and OmniCode exclusions retain their original partitions.
The full 117 validation attempts belong to the same 50 held-out tasks; the pilot's
50-record validation file remains unchanged.

`provenance.json` pins the tokenizer revision/template/files and source report;
per-part receipts permit checksum-validated resumption. `report.json` summarizes
counts and untruncated lengths; `verification_report.json` records full replay
verification. Dataset artifacts remain git-ignored.

Read a record without decompressing the entire corpus:

```python
import gzip
import json
from pathlib import Path

root = Path("data/processed/nvidia_mini_swe_agent_v1_2_success_qwen35_4b")
with gzip.open(next(root.glob("part-*.jsonl.gz")), "rt", encoding="utf-8") as f:
    record = json.loads(next(f))
print(record["partition"], record["token_count"])
print(record["text"])
```

### Rule-selected baseline: 4,000 tasks / trajectories

Run `.venv/bin/python -m scripts.data.select_rule_trajectories --workers 4`.
The full-corpus conversion must already have a matching `verification_report.json`.
The CPU-only selector caches compact features under
`data/interim/nvidia_success_rule_features_v1/`; no tokenization, model inference,
NLL scoring, or command execution occurs. Re-running verifies source checksums and
reuses the index. Add `--verify` to reread every source record, recompute all
candidate features (including unselected attempts), reproduce selection and compare
all report fields as well as exported payloads.

Outputs are under `data/processed/mini_swe_agent_rule_baseline_4000/`:

- `part-*.jsonl.gz`: exactly one unchanged Qwen-rendered record per selected task;
  all records belong to `train_pool`. Existing pilot/validation files are untouched.
- `task_selection_manifest.json`: 4,000 source locators, original payload checksums,
  feature scores, seed 42, proportional category quotas and pinned provenance.
- `task_candidate_rankings.jsonl`: valid/invalid candidates and lexicographic scores
  for every eligible task, with its winner and whether it was sampled for train.
- `candidate_exclusions.jsonl`, `task_exclusions.jsonl`: explicit rejection reasons.
- `remaining_task_winners.jsonl`: winners for eligible tasks outside the 4,000 sample.
- `report.json`, `verification_report.json`: coverage, untruncated lengths, historical
  shell-error/repetition counts, payload integrity and reproducibility checks.

Eligibility requires >=2 original successful attempts of a train-pool task and
>=1 valid candidate. Serialization failures, schema-invalid arguments, unsupported
categories, invalid token counts, and conversations with no tool calls are excluded.
Pick the minimum tuple `(bash_usage_errors, redundant_repeats, token_count,
tool_call_count, trajectory_id)` within each task. A shell usage error requires both
an explicit shell diagnostic and matching return code: command not found (127) or
shell syntax error (2). Other nonzero exits and unparseable observations are recorded
separately, never treated as automatic errors. Parse observation JSON once; commands
and observation text are never rewritten or repeatedly unescaped.

Consecutive duplicate blocks have identical function/arguments and original
observations; call IDs and reasoning are ignored, and an intervening different
operation (such as EDIT) breaks the repetition. Positional observation links are
allowed; a missing final observation for terminal submit is not a penalty.

Category quotas use largest remainder on counts of eligible canonical TASKS.
Within each category, seed 42 shuffles repositories and their task queues, then
samples repositories in round robin. This rule is a quality/complexity baseline,
not a measurement of student capability alignment. One epoch is the current training
intent; selected token totals still need remote QLoRA profiling for the 30h budget.
No labels, loss weights, truncation or training are implemented by this selector.
