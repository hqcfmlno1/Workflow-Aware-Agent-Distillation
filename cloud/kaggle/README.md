# Training Kaggle qua KRun

KRun chạy ở máy local; workload training chạy trong một job Kaggle. Thư mục này dành cho hướng dẫn và tài nguyên riêng của Kaggle. Project root để đóng gói là thư mục gốc repository, không phải `cloud/kaggle/`, để giữ được imports trong `src/`, configs và script entrypoint.

## Chuẩn bị

1. Cài KRun theo README chính thức: https://github.com/nhminh107/KRun-PyProjectOnKaggle.
2. Chạy `krun login` và `krun doctor` trên local.
3. Chốt dependency training tương thích Qwen3.5 và CUDA của phiên Kaggle. Repo có optional extra `train`; cần kiểm tra cách cài extras, không chỉ dựa vào việc phát hiện pyproject.toml.
4. Gắn dữ liệu lớn qua Kaggle Dataset; code đọc từ `/kaggle/input/`. Giới hạn đóng gói source của KRun là 20 MB trước nén.
5. Khi đã có training entrypoint, chạy KRun `--dry-run` trước để kiểm tra file đóng gói. `--dry-run` không thực thi training hoặc kiểm tra CUDA.
6. Pilot 50–100 bước, ghi số GPU thực cấp, tên GPU, peak VRAM, tốc độ, validation loss, lưu/resume.

## Dùng T4 ×2

Script phải khởi chạy distributed training qua torchrun/Accelerate khi dùng DDP; KRun không tự làm bước này. DDP giữ một bản model trên mỗi GPU và không gộp VRAM. Kiểm tra `torch.cuda.device_count()` trong job, không suy ra từ lựa chọn trên giao diện Kaggle.

## Kết quả và evaluation

Ghi kết quả vào `outputs/training/<run_id>/` để KRun thu thập. Job mới không được giả định giữ cache/checkpoint của job trước.

Validation loss có thể chạy trong job training. Evaluation mini-SWE-agent + OmniCode chạy trên máy hỗ trợ Docker với image cache; chưa xác minh Docker harness chạy được trong môi trường Kaggle. Evaluation còn cần inference endpoint của student tương thích với mini-SWE-agent.

Hiện repository mới có khung thư mục và manifest dữ liệu; chưa có training entrypoint, launcher multi-GPU hoặc cấu hình KRun chạy hoàn chỉnh.
