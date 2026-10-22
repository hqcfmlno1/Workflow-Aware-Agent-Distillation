# Workflow-Aware-Agent-Distillation

Baseline **Action-focused SFT** cho student **Qwen3.5-4B**, dùng successful
trajectories của teacher **Qwen3.8-27B Thinking** từ **NVIDIA Open-SWE-Traces
v1.2 / mini-SWE-agent**. Loss lấy ý tưởng từ SmartAD: weighted-token CE cho
reasoning/content/action/submit với trọng số **1 / 0.75 / 1.5 / 2**, rồi trung
bình đều theo trajectory. Mục tiêu evaluation là **SWE-bench Verified**.

## Trạng thái hiện tại

- Training đã chạy trên Kaggle và Vast. Bằng chứng pilot/resume 4080 được ghi
  trong repo ở checkpoint **1028 / 2.692 trajectory**, với Efficient SDPA và
  hybrid activation storage 4 GiB/GPU. Đây là mốc đã xác minh trong tài liệu,
  không phải tiến độ realtime; xem [profile 4080](configs/training/gpus/rtx4080_32gb/README.md).
- Đã chuẩn bị cohort ưu tiên đa dạng đến batch **2.000 / 4.000 trajectory**:
  **3.866 task**, **1.501 repo**. Giữ nguyên prefix đã học; thêm 2.610 trajectory
  từ các task mới, mỗi trajectory bổ sung dưới 28.000 token.
- Cohort đã được tích hợp trên Vast và mở rộng sang pool còn lại, giữ nguyên
  prefix đã học. Global batch hiện dùng 4 (2 trajectory/GPU).
- Luồng refactor local có một entrypoint [`scripts/train.py`](scripts/train.py),
  module [`training/`](src/workflow_distillation/training/) và profile riêng theo GPU.
  **Chưa triển khai bản refactor lên server; cần CUDA migration pilot trước khi dùng.**
  Bắt đầu review tại [bản đồ training](docs/training.md).
- Chưa hoàn tất epoch hoặc có kết quả evaluation student.

## Cấu trúc

```text
configs/                    # Manifest nguồn, bộ lọc, cấu hình thí nghiệm
scripts/
  train.py                  # Entrypoint refactor: preflight / launch / worker
  data/                     # Chuẩn bị dữ liệu và lịch chọn trajectory trên CPU
  training/                 # Pilot, tải checkpoint và launcher tương thích lịch sử
  cloud/                    # Launcher và phục hồi Kaggle
  evaluation/               # Dành cho entrypoint evaluation
src/workflow_distillation/
  training/                 # Pipeline refactor: config/data/model/engine/checkpoint
  *.py                      # Primitive dùng chung và implementation lịch sử đã pin hash
configs/training/gpus/      # Mỗi GPU một profile.json; xem README từng GPU
legacy/first_run/           # Inventory hash; source lịch sử giữ ở đường dẫn gốc
cloud/kaggle/               # Bootstrap/recovery và lịch continuation hiện có
data/
  raw/                      # Cache nguồn và tokenizer
  interim/                  # Export và dữ liệu trung gian
  processed/
    first_run_epoch_sft/    # index.json + tokens.bin + groups.bin gốc
    first_run_diverse_4000/ # Manifest cohort đến batch 2.000; tham chiếu binary gốc
    first_run_diverse_remaining/ # Pool mở rộng, bảo toàn cohort/prefix đã học
outputs/
  training/first run/       # Log, metrics, checkpoint và payload triển khai phát sinh
  evaluation/               # Kết quả benchmark
  analysis/                 # Phân tích phát sinh
reports/                    # Bằng chứng nghiên cứu/overlap được lưu trong Git
notebooks/                  # Phân tích NVIDIA trajectories
```

**Đầu vào train nằm trong `data/`; kết quả chạy nằm trong `outputs/`.** Manifest
cohort chỉ chọn thứ tự record từ binary gốc, không nhân bản dữ liệu. Những bộ
processed cũ được giữ làm nguồn tái lập; xem [danh mục dữ liệu](data/README.md).
Artifact lớn trong `data/` và `outputs/` được Git bỏ qua, trừ README.

## Chuẩn bị cohort hiện tại

```bash
uv sync --extra data
uv run python -m scripts.data.prepare_diverse_continuation \
  --index data/processed/first_run_epoch_sft/index.json \
  --source-plan cloud/kaggle/recovery/continuation-plan.json \
  --resume-step 695 --target-step 2000 --verify-binaries \
  --output data/processed/first_run_diverse_4000
```

Lệnh chỉ chuẩn bị/kiểm tra manifest trên CPU. Mốc resume phải được đối chiếu
với checkpoint thật trước khi train; nếu checkpoint mới hơn, tạo cohort mới
giữ toàn bộ prefix đã học. Không sửa index, binary hay lịch continuation gốc.

## Nguồn và chống trùng benchmark

[Manifest nguồn](configs/datasets/nvidia_mini_swe_agent_v1_2_action_focused_sft.json)
pin dataset/revision, teacher/harness và quy tắc partition. Loại task trùng
benchmark theo repository + PR chuẩn hóa trước khi chuẩn bị train; không đưa
reference patch vào đầu vào student. Giữ các
[báo cáo OmniCode](docs/data_overlap_v1_2_2026-10-11.md) và
[audit SWE-bench Verified](reports/swebench_verified_overlap_2026-10-12/)
để tái lập bộ lọc.

Collector OmniCode/OpenAI, script phụ trợ, Dockerfile và hai notebook phân tích
collector đã được gỡ vì không còn dùng. Các phiên bản cũ vẫn có trong lịch sử Git.
OmniCode task keys phục vụ chống trùng vẫn được giữ.

Xem thêm [scripts](scripts/README.md), [output](outputs/README.md),
[thiết kế loss](docs/action_focused_sft_loss_design.md) và
[pipeline chuẩn bị ban đầu](docs/action_focused_sft_phase1.md).
