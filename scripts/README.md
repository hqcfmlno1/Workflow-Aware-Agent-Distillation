# Entrypoint

Các script collector SWE-agent hiện có ở thư mục này được giữ nguyên. Pipeline mới dùng mini-SWE-agent; adapter evaluation mới chưa được triển khai.

Các thư mục dành cho pipeline mới:

- `data/`: tải dữ liệu, lọc teacher/harness/category, loại task trùng và chia dataset.
- `training/`: pilot QLoRA, training và resume checkpoint.
- `evaluation/`: validation loss và chạy evaluation mini-SWE-agent/OmniCode.
- `cloud/`: chuẩn bị payload và khởi chạy batch job qua KRun.

Entrypoint dữ liệu: `python scripts/data/prepare_cold_start.py` (CPU, tokenizer only).
Xem [pipeline phase 1 và metric](../docs/cold_start_phase1.md). Các entrypoint training,
evaluation và cloud chưa được triển khai; máy local không chịu trách nhiệm fine-tune.
