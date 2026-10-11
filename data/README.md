# Dữ liệu cục bộ

Máy local chỉ chuẩn bị dữ liệu/tokenizer trên CPU. Fine-tune chạy ở môi trường khác.
Xem [pipeline và metric phase 1](../docs/cold_start_phase1.md).

| Thư mục | Nội dung |
| --- | --- |
| `raw/nvidia_open_swe_traces/` | Cache cột nguồn NVIDIA theo revision; không đọc reference patch |
| `raw/qwen35_4b_tokenizer/` | Tokenizer/chat template chính thức; không tải model weights |
| `interim/` | Kết quả lọc và chuẩn hóa trung gian |
| `processed/mini_swe_agent_v1_2_phase1/` | 1.000 train + 100 validation, text, pool, checks và manifest |
| `splits/` | Dành cho các split khác sau này; phase 1 lưu split cùng processed artifacts |
| `manifests/` | Dành cho các lần chuẩn bị khác; manifest phase 1 nằm cùng processed artifacts |

Bắt đầu từ `configs/datasets/nvidia_mini_swe_agent_v1_2_bugfix.json`. Loại task trùng OmniCode trước khi chia train/validation; tất cả trajectory cùng repository + PR đã chuẩn hóa phải cùng split, kể cả khi instance_id khác định dạng giữa hai nguồn. Không đưa reference_patch vào đầu vào student.

Dữ liệu phát sinh không được commit. Các thư mục rỗng có thể không xuất hiện sau clone; bước chuẩn bị dữ liệu cần tạo chúng. Với Kaggle, gắn dataset lớn qua Kaggle Dataset dưới `/kaggle/input/`; cấu hình training phải nhận đường dẫn đầu vào thay vì giả định đường dẫn máy local.
