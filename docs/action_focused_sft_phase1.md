# Action-focused SFT — Phase 1

Phase 1 tinh chỉnh một student đã có năng lực agent, ưu tiên tool calls, Bash commands
và hành vi submit từ teacher trajectories. Tên Action-focused SFT mô tả mục tiêu này.
Thiết kế loss nằm trong [tài liệu hàm loss](action_focused_sft_loss_design.md).

Máy local chỉ chuẩn bị dữ liệu trên CPU và lưu tokenizer/chat template; fine-tune chạy
ở môi trường khác (dự kiến Kaggle). Pipeline không tải weights, khởi tạo model, dùng GPU,
train/evaluate student hoặc thực thi command teacher. Collector SWE-agent cũ giữ nguyên.

## Chạy

```bash
uv sync --extra data
uv run python scripts/data/prepare_action_focused_sft.py
uv run python scripts/data/prepare_action_focused_sft.py --offline
uv run python scripts/data/prepare_action_focused_sft.py --verify
```

Extra `data` chứa transformers/tokenizers, Jinja2, jsonschema; không cần extra `train`.
Offline cần cache đầy đủ. Output đã có `report.json` không bị ghi đè; đổi
`local_paths.processed` trong config để chạy mới. Lần chưa hoàn tất dùng tiếp cache được.

Manifest `nvidia_mini_swe_agent_v1_2_action_focused_sft.json` có selection `phase1` riêng:
250 train + 25 held-out cho mỗi category bug-fix, feature-request, enhancement, other.
Giữ 50 validation theo seed 42, có lần lượt 13/13/12/12 mẫu theo thứ tự category trên.
50 task còn lại từ reserve cũ được trả về training pool. Train pilot 1.000 giữ nguyên.
Không có test nội bộ; test cuối dùng 500 task SWE-bench Verified theo config.
Teacher Qwen3.8-27B Thinking, resolved=1, revision NVIDIA đã audit, đủ 44 shard.
Sắp xếp trước khi shuffle repository/task/attempt bằng seed 42; luân phiên repository,
held-out trước, một trajectory mỗi task. Sau đó sort task key và shuffle theo seed 42
trong từng category để chọn validation; phần còn lại trở về pool. Thiếu quota thì dừng, không đổi phân bố.

Đọc đủ 494 base + 415 style OmniCode, checksum sáu file phải khớp audit. Khóa task là
repo viết thường + số PR; style dùng original_instance_id/pull_number để tránh nhầm số
biến thể. Không khẳng định phát hiện các PR khác nhau giải cùng issue.

## Artifacts

Trong `data/processed/mini_swe_agent_v1_2_phase1/` (Git bỏ qua):

| File | Nội dung |
| --- | --- |
| train.jsonl, validation.jsonl | Normalized messages/tools, text, token count, checks, provenance, action candidates |
| task_selection_manifest.json | Config/seed, task IDs, benchmark hashes, tokenizer commit/template/file hashes |
| excluded.jsonl | Loại trước sampling và lý do |
| errors.jsonl | Ứng viên không thể normalize/serialize, thay trong cùng category |
| checks.jsonl | Call sai schema và trường hợp chưa xác minh |
| task_pool.jsonl | Mọi attempts không trùng benchmark; cùng task giữ cùng partition |
| report.json | Coverage category/repo/source/ngôn ngữ/thao tác, độ dài, checks, artifact hashes |
| verification_report.json | Kiểm chứng offline với source, template, token count, spans và partition pool |
| failure.json | Lý do thiếu quota nếu có |

Phase 2 phải loại mọi task có partition=validation, kể cả attempts chưa chọn hoặc
unsuccessful/khác category. Pool chưa phải train: cần lọc teacher/category/resolved và
kiểm tra hội thoại khi sử dụng.

Chỉ đọc provenance và metadata category/teacher, không đọc reference/model patches
NVIDIA. Giữ content, reasoning, mọi tool calls và observations; decode một lớp JSON cho
tools và arguments, không unescape command thêm. Cache giữ source messages. Normalized
giữ reasoning ngay cả khi template bỏ lịch sử reasoning. Template Qwen3.5-4B chính thức
dùng add_generation_prompt=False; không truncate, packing hoặc tạo labels.

Call parse được nhưng sai schema được giữ với check invalid: teacher có thể gọi sai
rồi sửa. JSON không normalize được hoặc trình tự observation sai sẽ loại trajectory.
Observation thiếu ID chỉ đối chiếu vị trí và ghi unverified. Cho phép terminal submit
cuối không có observation. Không suy ra command hợp lệ từ resolved=1.

Action candidates là chỉ số ký tự Python [start,end) trong text và message/call index.
Marker render phân biệt call thật với tag trong reasoning/content/observation; bỏ marker
phải khớp tuyệt đối render chính thức. Đối số cuối không phải string có span null và
trạng thái chưa xác minh. Candidates chưa phải labels hay bằng chứng command chạy được.

## Metric so base student với LoRA sau này

Đo trên output student với cùng harness, prompt, tool schema, image/dependency state,
task split, generation config và budget. Ghi revision model/adapter/parser và image digest.
Lần chuẩn bị dữ liệu chưa đo các metric này.

**Tool-call validity:** output qua parser harness theo protocol Qwen, mọi call đúng tên
tool và JSON schema ngay lần đầu, không retry/sửa JSON/fallback. Mẫu số: mọi lần sinh
được yêu cầu action; không parse được hoặc thiếu call là invalid. Báo cáo cả tỷ lệ theo
call và theo lượt (multi-call chỉ đạt khi tất cả calls hợp lệ). Final response hợp lệ
theo harness báo cáo riêng, không tùy ý thêm vào mẫu số action.

**Command availability:** trong Bash calls hợp lệ, từng chương trình/builtin được gọi
phải được xác nhận tồn tại trong environment harness ở đúng shell/PATH/cwd/task/setup
state tại thời điểm đó. Không dùng PATH máy local hay resolved=1 để suy ra availability.
Command ghép cần kiểm tra từng invocation; một invocation xác nhận thiếu khiến call
unavailable. Ba trạng thái: verified_available, verified_unavailable, unverified.
Báo cáo cả ba số lượng trên tổng calls hợp lệ, tỷ lệ available trên phần đã xác minh
và coverage xác minh để tránh bỏ trường hợp khó làm đẹp tỷ lệ.

Command động, substitutions, scripts chưa có nội dung, PATH thay đổi hoặc dependencies/
environment chưa có là unverified. Availability không đồng nghĩa exit code 0 hoặc giải
đúng task. Phân loại thao tác trong report chỉ là heuristic chương trình đầu tiên,
không phải metric availability. Dataset qua kiểm tra chưa chứng minh LoRA đạt mục tiêu.

## Test cuối và dữ liệu lịch sử

Test cuối là 500 task `princeton-nlp/SWE-bench_Verified`, revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Config `test_evaluation` ghi dataset,
revision, split và audit. Không dùng test để chọn checkpoint hoặc hyperparameter.
Audit `reports/swebench_verified_overlap_2026-10-12/` xác nhận không trùng repository
+ PR với 107.267 trajectories nguồn; chưa kiểm semantic duplicates hoặc issue chung.

Validation hiện tại giữ nguyên 50 task đã chọn. Mọi attempts của 50 task từng thuộc
test nội bộ đã trở về pool. `test.jsonl` cũ, manifest/report và success export cũ được
lưu trong `data/interim/archive_internal_test_2026-10-12/`, ngoài các đường dẫn đang dùng.
Train/validation processed vẫn là trajectory đầy đủ, chưa tạo next-action examples
hoặc labels. Export success hiện hành đánh dấu `train_pool`, `validation` và
`excluded_omnicode`. Chuẩn bị train toàn bộ success pool là bước riêng, chưa thực hiện.

## Full-corpus template conversion

The separate CPU-only `scripts/data/convert_success_trajectories.py` pipeline
renders every success-export row using the same pinned official Qwen3.5-4B
tokenizer as the pilot. Its sharded gzip JSONL outputs are under
`data/processed/nvidia_mini_swe_agent_v1_2_success_qwen35_4b/`. It preserves task
partitions and all attempts, and does not change the 1,000+50 pilot selection.
Normalization failures remain as explicit error records with original payloads;
there is no silent replacement or dropping. Loss masks and SFT training labels
remain a subsequent trainer-specific step. See `data/README.md` for build, replay
verification, and streaming-read commands.

## Rule-selected SFT baseline (2026-10-13)

The current baseline selection is 4,000 distinct train-pool tasks with multiple
successful attempts, selecting one original Qwen-rendered trajectory per task.
`scripts/data/select_rule_trajectories.py` ranks candidates lexicographically by
confirmed historical shell usage errors, consecutive duplicate action/observation
blocks, serialized token count, tool-call count, and trajectory ID. It excludes
schema-invalid calls and keeps held-out tasks out of all sampling. Category quotas
are proportional to eligible tasks; repository sampling is seeded round robin.

Outputs, candidate diagnostics, remaining winners and full verification are under
`data/processed/mini_swe_agent_rule_baseline_4000/`; see `data/README.md` for details.
This changes no pilot or validation record. Training is planned for one epoch with
action-focused supervision. The group-normalized weighted CE design is documented
in `action_focused_sft_loss_design.md`; loss masks, trainer implementation, remote
profiling, and training remain separate work. Rules are a quality/complexity proxy,
not student NLL alignment.
