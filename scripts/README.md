# Entrypoint

Các script collector SWE-agent hiện có ở thư mục này được giữ nguyên. Pipeline mới dùng mini-SWE-agent; adapter evaluation mới chưa được triển khai.

Các thư mục dành cho pipeline mới:

- `data/`: tải dữ liệu, lọc teacher/harness/category, loại task trùng và chia dataset.
- `training/`: pilot QLoRA, training và resume checkpoint.
- `evaluation/`: validation loss và chạy evaluation mini-SWE-agent/OmniCode.
- `cloud/`: chuẩn bị payload và khởi chạy batch job qua KRun.

Entrypoint dữ liệu: `python scripts/data/prepare_action_focused_sft.py` (CPU, tokenizer only).
Xem [pipeline phase 1 và metric](../docs/action_focused_sft_phase1.md). Trainer baseline
và evaluation chưa hoàn chỉnh; máy local không chịu trách nhiệm fine-tune.
Các pilot QLoRA chạy remote qua KRun đã có checkpoint định kỳ và resume; xem
[hướng dẫn checkpoint](../docs/training_checkpoint_resume.md). Công cụ CPU
`training/fetch_training_checkpoint.py` tải bản mới nhất từ repo HF private để
gắn vào phiên Kaggle tiếp theo.
