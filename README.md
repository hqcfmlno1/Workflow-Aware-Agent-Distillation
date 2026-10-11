# Workflow-Aware-Agent-Distillation

Distill student **Qwen3.5-4B** từ teacher **Qwen3.8-27B Thinking** trên **NVIDIA Open-SWE-Traces v1.2**, sử dụng **mini-SWE-agent** harness. Training dự kiến chạy QLoRA trên GPU Kaggle qua KRun; evaluation dùng OmniCode, bắt đầu với workflow bugfix.

## Cấu trúc

```text
configs/
  datasets/       # Nguồn, revision, bộ lọc và chống trùng benchmark
  training/       # Cấu hình student và QLoRA
  harness/        # mini-SWE-agent prompt, actions và model connection
  evaluation/     # Task selection, budget và verifier
scripts/
  data/           # Chuẩn bị và chia NVIDIA trajectories
  training/       # Pilot, training và resume
  evaluation/     # Validation và evaluation agent
  cloud/          # Launcher KRun
  ...             # Collector và utility hiện có
src/workflow_distillation/   # Logic Python tái sử dụng
cloud/kaggle/                # Hướng dẫn workload Kaggle/KRun
data/
  raw/nvidia_open_swe_traces/
  interim/
  processed/mini_swe_agent_v1_2_phase1/
  splits/
  manifests/
outputs/
  training/<run_id>/
  evaluation/<run_id>/
  analysis/<run_id>/
reports/                    # Báo cáo và bằng chứng nghiên cứu được lưu
notebooks/                  # Phân tích khám phá
external/                   # Checkout OmniCode/mini-SWE-agent cục bộ, khi cần
```

`data/`, `outputs/` và `external/` chứa artifact cục bộ, được bỏ qua bởi Git trừ README hướng dẫn. Các thư mục rỗng cho dữ liệu/output có thể không tồn tại sau clone. Collector hiện có dùng SWE-agent được giữ nguyên để tham khảo; chưa chuyển sang mini-SWE-agent.

## Nguồn và chống trùng evaluation

Manifest: [configs/datasets/nvidia_mini_swe_agent_v1_2_bugfix.json](configs/datasets/nvidia_mini_swe_agent_v1_2_bugfix.json).

Chọn `nvidia/Open-SWE-Traces`, config `v1.2`, split `minisweagent`, teacher Qwen3.8-27B Thinking và `resolved == 1`. Phase 1 lấy 250 train + 25 validation cho mỗi category `bug-fix`, `feature-request`, `enhancement`, `other`, từ cả Scale-SWE và SWE-rebench-v2. Giữ reasoning/action/observation và serialize bằng template student chính thức. Chưa cắt context, packing hay tạo training labels.

Loại task trùng OmniCode trước khi chia train/validation. Gom theo repository + số PR đã chuẩn hóa để các ID khác nhau của cùng task giữa hai nguồn không bị chia sang hai tập. Xem [overlap v1.2](docs/data_overlap_v1_2_2026-10-11.md) và [exclusion manifest v1.2](reports/data_overlap_2026-10-11_v1_2/training_exclusions.json). Báo cáo ngày 2026-10-10 được giữ làm bằng chứng lịch sử của snapshot dataset.

## Trạng thái và thứ tự triển khai

Repo có pipeline chuẩn bị NVIDIA phase 1, collector OmniCode và báo cáo overlap.
Máy local chỉ chuẩn bị dữ liệu và lưu tokenizer; fine-tune chạy trên môi trường khác.
Trainer QLoRA, launcher KRun và evaluation student chưa được triển khai.
Xem [cách chạy và định nghĩa metric](docs/cold_start_phase1.md).

1. Chuẩn bị một tập NVIDIA nhỏ, loại overlap và chia train/validation.
2. Pilot Qwen3.5-4B QLoRA trên GPU Kaggle thực cấp; kiểm tra memory, tốc độ, reasoning serialization và resume.
3. Kiểm tra student qua mini-SWE-agent trên 5–10 task OmniCode với verifier. Adapter OmniCode cho harness mới cần được triển khai/kiểm chứng riêng.
4. Mở rộng training và evaluation sau khi hai bước trên chạy ổn.

Xem [cấu hình](configs/README.md), [scripts](scripts/README.md), [dữ liệu](data/README.md), [output](outputs/README.md) và [KRun/Kaggle](cloud/kaggle/README.md).
