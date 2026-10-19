# Kết quả chạy

Đầu vào train đã chốt nằm trong `data/processed/`: binary gốc tại
`first_run_epoch_sft/`, manifest cohort mới tại `first_run_diverse_4000/`.
`outputs/` dành cho kết quả thực thi; không dùng làm thư mục dataset chính.

- `training/<run_id>/`: config đã chốt, log, metrics, adapter/tokenizer và `checkpoints/`.
- `evaluation/<run_id>/`: task manifest, trajectories, patches, verifier results và metrics.
- `analysis/<run_id>/`: bảng/biểu đồ tổng hợp phát sinh.

Với run hiện tại, `training/first run/` còn chứa các bản payload triển khai,
metadata Kaggle và chẩn đoán sự cố đã phát sinh. Chúng là snapshot của từng
lần chạy, không phải mã nguồn hoặc dataset chính để sửa. Giữ chúng để tra cứu
sự cố; dùng source trong `scripts/`, `src/`, `cloud/` cho lần triển khai mới.
Checkpoint tải về máy local hiện ở `training/first run/local-checkpoints/`.

`run_id` nên gồm ngày, student và tên cấu hình để tránh ghi đè. Mỗi run lưu dataset revision, code revision, seed và cấu hình đã resolve. Resume training cần cả optimizer/scheduler/trainer state nếu muốn tiếp tục đúng bước.

KRun thu thập `outputs/` mặc định. Code chạy Kaggle phải ghi vào đúng thư mục này tương đối với project. Để tái sử dụng checkpoint trong job mới, cần tải/gắn checkpoint làm input và truyền đường dẫn resume.

Kết quả phát sinh không được commit. Báo cáo nghiên cứu được chọn để lưu lâu dài nằm trong `reports/`. Các thư mục rỗng có thể không xuất hiện sau clone; runner cần tạo thư mục run trước khi ghi.
