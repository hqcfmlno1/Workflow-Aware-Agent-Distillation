# Entrypoint

Chạy từ thư mục gốc repo, ưu tiên `python -m scripts.<nhóm>.<tên_script>`.
Máy local chuẩn bị dữ liệu trên CPU; fine-tune chạy remote.

| Thư mục | Trách nhiệm |
| --- | --- |
| `data/` | Tải/export NVIDIA, normalize, tokenize, lọc và chọn thứ tự trajectory |
| `training/` | Pilot QLoRA, epoch training, tải checkpoint HF |
| `cloud/` | Preflight và bootstrap/resume Kaggle |
| `evaluation/` | Dành cho evaluation student qua mini-SWE-agent |

## Luồng dữ liệu

1. `data/prepare_action_focused_sft.py`: cache nguồn, loại overlap và chuẩn bị pilot.
2. `data/export_success_trajectories.py`: export successful train/validation/excluded records.
3. `data/convert_success_trajectories.py`: normalize/render theo tokenizer Qwen.
4. `data/prepare_epoch_sft.py`: tạo `data/processed/first_run_epoch_sft/`
   với index và binary cho successful train-pool trajectories dưới 30.000 token.
5. `data/prepare_diverse_continuation.py`: tạo manifest cohort trong
   `data/processed/first_run_diverse_4000/`, giữ prefix đã học và chọn task/repo
   đa dạng từ suffix của lịch continuation `<28.000 token`.

`select_rule_trajectories.py`, `prepare_weighted_sft.py` và
`select_weighted_sft_context.py` phục vụ selection/pilot trước đây; chúng không
chọn cohort tiếp tục trên Vast. Xem [dữ liệu](../data/README.md) để phân biệt input.

## Training và checkpoint

`training/train_qlora_epoch.py` là trainer epoch gốc. Bootstrap
`cloud/resume_first_run.py` và `training/train_qlora_filtered.py` phục vụ
continuation Kaggle hiện có; giữ nguyên identity/lịch đã dùng để train.
Xem [first run](../docs/first_run_epoch.md) và
[checkpoint/resume](../docs/training_checkpoint_resume.md).

`training/fetch_training_checkpoint.py` tải checkpoint mới nhất từ HF.
Cohort đa dạng mới là artifact dữ liệu; **chưa có launcher Vast tích hợp cohort**.
Xem [điều kiện trước khi chạy Vast](../docs/vast_diversity_continuation.md).

Collector OmniCode/OpenAI và các utility riêng của collector đã được gỡ.
