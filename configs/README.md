# Cấu hình thí nghiệm

- `datasets/`: manifest nguồn NVIDIA, revision, teacher/harness, bộ lọc và chống trùng benchmark.
- `training/`: dành cho các cấu hình training; cấu hình QLoRA hiện tại được mô tả ở [qlora_current_config.md](../docs/qlora_current_config.md).
- `harness/`: cấu hình mini-SWE-agent dùng để chạy student; chốt tools và prompt tương thích dữ liệu.
- `evaluation/`: dành cho task selection, budget, timeout và verifier; mục tiêu hiện tại là SWE-bench Verified qua mini-SWE-agent.

`datasets/nvidia_mini_swe_agent_v1_2_action_focused_sft.json` là cấu hình cho
`scripts/data/prepare_action_focused_sft.py`, với selection `phase1` riêng
cho bốn category. Máy local chỉ chuẩn bị dữ liệu/tokenizer trên CPU; tuning chạy ở
môi trường khác. Manifest không truyền trực tiếp cho Transformers trainer, mini-SWE-agent
hay KRun. Xem [hướng dẫn](../docs/action_focused_sft_phase1.md).
