# Entrypoint

Training refactor bắt đầu tại `python scripts/train.py --help`.
Các script phụ trợ chạy từ gốc repo bằng `python -m scripts.<nhóm>.<tên_script>`.
Máy local chuẩn bị dữ liệu trên CPU; fine-tune chạy remote.

| Thư mục | Trách nhiệm |
| --- | --- |
| `data/` | Tải/export NVIDIA, normalize, tokenize, lọc và chọn thứ tự trajectory |
| `train.py` | Entrypoint refactor: validate/preflight/launch/worker |
| `training/` | Pilot QLoRA, tải checkpoint HF và launcher lịch sử |
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

Luồng local mới: `train.py` → `src/workflow_distillation/training/cli.py` →
`engine.py`, chọn GPU bằng `configs/training/gpus/<GPU>/profile.json`.
Xem [bản đồ review và migration](../docs/training.md). Bản refactor chưa deploy
lên server hoặc pilot trên CUDA; nó không tự thay launcher của run remote.

`training/train_qlora_epoch.py` là trainer epoch gốc. Bootstrap
`cloud/resume_first_run.py` và `training/train_qlora_filtered.py` phục vụ
continuation Kaggle hiện có; giữ nguyên identity/lịch đã dùng để train.
Xem [first run](../docs/first_run_epoch.md) và
[checkpoint/resume](../docs/training_checkpoint_resume.md).

`training/fetch_training_checkpoint.py` tải checkpoint mới nhất từ HF.
`training/train_qlora_vast*.py` và `training/gpus/rtx4080_32gb/train.py` là các
đường chạy đã dùng cho Vast trước refactor. Giữ nguyên chúng để bảo toàn hash
và resume; không thêm orchestration mới vào chain launcher này.
Xem [inventory tương thích](../legacy/README.md).

Collector OmniCode/OpenAI và các utility riêng của collector đã được gỡ.
