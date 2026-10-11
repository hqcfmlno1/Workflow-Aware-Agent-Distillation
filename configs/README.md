# Cấu hình thí nghiệm

- `datasets/`: manifest nguồn NVIDIA, revision, teacher/harness, bộ lọc và chống trùng benchmark.
- `training/`: cấu hình QLoRA student, context, batch, optimizer và distributed training khi triển khai trainer.
- `harness/`: cấu hình mini-SWE-agent dùng để chạy student; chốt tools và prompt tương thích dữ liệu.
- `evaluation/`: task selection, model budget, timeout và verifier của OmniCode.

`datasets/nvidia_mini_swe_agent_v1_2_bugfix.json` là cấu hình cho
`scripts/data/prepare_cold_start.py`, giữ tên lịch sử nhưng có selection `phase1` riêng
cho bốn category. Máy local chỉ chuẩn bị dữ liệu/tokenizer trên CPU; tuning chạy ở
môi trường khác. Manifest không truyền trực tiếp cho Transformers trainer, mini-SWE-agent
hay KRun. Xem [hướng dẫn](../docs/cold_start_phase1.md).
