# Kết quả chạy

- `training/<run_id>/`: config đã chốt, log, metrics, adapter/tokenizer và `checkpoints/`.
- `evaluation/<run_id>/`: task manifest, trajectories, patches, verifier results và metrics.
- `analysis/<run_id>/`: bảng/biểu đồ tổng hợp phát sinh.

`run_id` nên gồm ngày, student và tên cấu hình để tránh ghi đè. Mỗi run lưu dataset revision, code revision, seed và cấu hình đã resolve. Resume training cần cả optimizer/scheduler/trainer state nếu muốn tiếp tục đúng bước.

KRun thu thập `outputs/` mặc định. Code chạy Kaggle phải ghi vào đúng thư mục này tương đối với project. Để tái sử dụng checkpoint trong job mới, cần tải/gắn checkpoint làm input và truyền đường dẫn resume.

Kết quả phát sinh không được commit. Báo cáo nghiên cứu được chọn để lưu lâu dài nằm trong `reports/`. Các thư mục rỗng có thể không xuất hiện sau clone; runner cần tạo thư mục run trước khi ghi.
