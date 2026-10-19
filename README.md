# Workflow-Aware-Agent-Distillation

Baseline **Action-focused SFT** cho student **Qwen3.5-4B**, dùng successful
trajectories của teacher **Qwen3.8-27B Thinking** từ **NVIDIA Open-SWE-Traces
v1.2 / mini-SWE-agent**. Loss lấy ý tưởng từ SmartAD: weighted-token CE cho
reasoning/content/action/submit với trọng số **1 / 0.75 / 1.5 / 2**, rồi trung
bình đều theo trajectory. Mục tiêu evaluation là **SWE-bench Verified**.

## Trạng thái hiện tại

- QLoRA đã train trên Kaggle; mốc người dùng xác nhận gần nhất là checkpoint
  **695**, tương ứng **1.390 trajectory**. Xem [cấu hình hiện tại](docs/qlora_current_config.md)
  và [training/checkpoint trên Kaggle](docs/first_run_epoch.md).
- Đã chuẩn bị cohort ưu tiên đa dạng đến batch **2.000 / 4.000 trajectory**:
  **3.866 task**, **1.501 repo**. Giữ nguyên prefix đã học; thêm 2.610 trajectory
  từ các task mới, mỗi trajectory bổ sung dưới 28.000 token.
- Chuyển sang Vast đang trong bước chuẩn bị. Cohort mới **chưa được tích hợp
  để resume/train trên Vast**; cần xác minh checkpoint HF thực tế, chuyển đổi
  identity có kiểm soát và pilot 3090. Xem [hướng dẫn Vast](docs/vast_diversity_continuation.md).
- Chưa hoàn tất epoch hoặc có kết quả evaluation student.

## Cấu trúc

```text
configs/                    # Manifest nguồn, bộ lọc, cấu hình thí nghiệm
scripts/
  data/                     # Chuẩn bị dữ liệu và lịch chọn trajectory trên CPU
  training/                 # Pilot, training, tải checkpoint
  cloud/                    # Launcher và phục hồi Kaggle
  evaluation/               # Dành cho entrypoint evaluation
src/workflow_distillation/  # Dataset, loss, QLoRA runtime, checkpoint, selection
cloud/kaggle/               # Bootstrap/recovery và lịch continuation hiện có
data/
  raw/                      # Cache nguồn và tokenizer
  interim/                  # Export và dữ liệu trung gian
  processed/
    first_run_epoch_sft/    # index.json + tokens.bin + groups.bin gốc
    first_run_diverse_4000/ # Manifest cohort đến batch 2.000; tham chiếu binary gốc
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
