# Cấu hình nghiên cứu và overlap NVIDIA v1.2

Ngày cập nhật: 2026-10-11.

- Student: `Qwen/Qwen3.5-4B`.
- Teacher: `Qwen3.8-27B`, Thinking bật trong metadata.
- Harness: mini-SWE-agent; split dataset là `minisweagent`.
- Main data: `nvidia/Open-SWE-Traces`, config `v1.2`.
- Training dự kiến: QLoRA trên Kaggle qua KRun.
- Evaluation dự kiến: OmniCode, bắt đầu với bugfix; adapter mini-SWE-agent/OmniCode chưa được triển khai.

## Kiểm chứng nguồn

Giữ revision NVIDIA `e192b26a43b8d09f156eeee781aef8e6bd5108ba`, đã được audit toàn bộ ngày 2026-10-10. Config v1.2 chọn `data/minisweagent/qwen38_27b/*/*.parquet`, bao gồm cả Scale-SWE và SWE-rebench-v2.

Đã đọc metadata của đủ 44 shard, tổng 107.267 trajectory. Cả 107.267 đều có `metadata.teacher_model.name == Qwen3.8-27B` và `enable_thinking == true`. Đây là xác nhận chế độ sinh theo metadata, chưa phải kiểm tra mọi message đều có reasoning không rỗng hoặc reasoning đã serialize đúng cho student.

Phân bố category: 54.177 bug-fix, 32.365 feature-request, 13.384 enhancement, 7.341 other. Bộ lọc `resolved == 1`, category bug-fix, Thinking true cho 26.770 trajectory trước loại trùng benchmark.

## Overlap OmniCode

Dùng benchmark revision `87de6c4e0ecdb132bfcba96ee2e0a624ce6f26ad`, đối chiếu 494 base task và 415 style task theo repository + PR chuẩn hóa. Trích kết quả đúng teacher/harness từ [audit toàn bộ](data_overlap_audit_2026-10-10.md), bổ sung kiểm tra category và teacher metadata trên v1.2.

| Task gốc OmniCode | Số trajectory v1.2 trùng | Số còn trong successful bugfix |
| --- | ---: | ---: |
| `scrapy__scrapy-6050` | 2 | 2 |
| `scrapy__scrapy-6469` | 2 | 0 |
| `simdjson__simdjson_2016` | 3 | 0 |

Với simdjson, ID NVIDIA là `simdjson__simdjson-2016`; không dùng phép so sánh nguyên chuỗi ID để loại trùng. Không tìm thấy exact repository/PR overlap với tập style.

Sau khi loại hai trajectory bugfix thành công của Scrapy PR 6050, còn **26.768 trajectory** trước các bước lọc độ dài, dedup, kiểm tra message và chia train/validation. Đây là số trajectory, không phải số task duy nhất hay số mẫu training sau chia theo lượt.

Hai trajectory phải loại ở phạm vi successful bugfix:

- `c502bc32-4615-4e80-830f-2931994162c9`
- `ac1c0f65-863a-4dc0-be0e-2d031231f212`

Loại trùng trước khi chia dataset. Gom trajectory theo khóa repository + PR chuẩn hóa để tránh cùng task xuất hiện ở hai nguồn với ID khác định dạng. Nếu thay đổi category hoặc bổ sung unsuccessful trajectory, dùng toàn bộ ba benchmark keys, không chỉ hai trajectory ID trên.

## Artifacts và giới hạn

- Manifest hoạt động: `configs/datasets/nvidia_mini_swe_agent_v1_2_action_focused_sft.json`.
- Báo cáo số đếm: `reports/data_overlap_2026-10-11_v1_2/summary.json`.
- Loại trừ: `reports/data_overlap_2026-10-11_v1_2/training_exclusions.json`.
- Chi tiết bảy trajectory: `reports/data_overlap_2026-10-11_v1_2/overlapping_trajectories.csv`.
- Metadata cột đã đọc được giữ cục bộ tại `/tmp/swe-overlap-audit/v12_metadata/`.

Phạm vi exact repository/PR không phát hiện đầy đủ hai PR khác nhau giải cùng issue, fork hay repository đổi tên. Audit patch ngày 2026-10-10 kiểm tra mẫu của từng PR; cập nhật này không tuyên bố kiểm chứng patch của mọi trajectory v1.2. Không suy luận về contamination trong pretraining của student.

Collector cũ trong `src/workflow_distillation/omnicode_collector.py` vẫn dùng SWE-agent. Thay đổi manifest không tự đổi runner hoặc prompt/tool protocol của collector đó. Cần triển khai đường chạy mini-SWE-agent và giữ cùng harness khi so sánh student gốc với student sau distillation.
