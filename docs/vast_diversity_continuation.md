# Lịch ưu tiên đa dạng đến batch 2.000 trên Vast

Mục tiêu được chốt ngày 2026-10-19: tổng **4.000 trajectory** tại optimizer
batch **2.000**, với global batch size 2 (một trajectory/GPU). Mốc đã học dùng
để chuẩn bị là checkpoint **695**, tương ứng **1.390 trajectory**. Bộ chọn chỉ
chuẩn bị lịch trên CPU; không khởi chạy GPU, sửa checkpoint hoặc ghi lên HF.

## Cách chọn

- Đọc index token/group gốc và lịch continuation `<28.000 token` hiện có.
  Kiểm index hash, khôi phục lịch từ seed/boundary gốc và đối chiếu thứ tự.
- Giữ nguyên toàn bộ prefix 1.390 trajectory, kể cả task lặp và trajectory dài
  đã học. Không áp dụng lại bộ lọc tại checkpoint 695.
- Trong phần chưa học, bỏ khỏi tập ứng viên các task đã xuất hiện trong prefix.
  Với mỗi task mới, lấy trajectory đủ điều kiện xuất hiện sớm nhất trong lịch cũ.
- Mỗi lần chọn từ repo có **ít trajectory nhất trong toàn bộ cohort đã chọn**,
  tính cả prefix. Nếu bằng nhau, ưu tiên thứ tự ứng viên trong lịch gốc. Đây là
  phép chọn greedy, xác định được hoàn toàn, không thêm shuffle ngẫu nhiên.
- Chọn thêm đúng 2.610 trajectory, mỗi trajectory thuộc một task mới khác nhau.
  Nếu không đủ task mới, báo lỗi thay vì âm thầm bù bằng task lặp.
- Task/instance được đếm theo khóa chuẩn hóa `owner/repo#PR`; repo được đếm
  theo phần `owner/repo`. Đây là độ phủ task/repo, không phải phép đo độ đa dạng
  ngữ nghĩa hay bảo đảm chất lượng benchmark.

Không cắt context, ghi lại token/group binary, thay loss weights, learning rate,
LoRA rank, optimizer hay batch size. Những trajectory chưa chọn vẫn còn trong
dữ liệu nguồn; cohort này chỉ xác định mục tiêu 4.000 trajectory hiện tại.

## Kết quả kiểm trên dữ liệu local

| Chỉ số | Đã học tại 695 | Lịch cũ đến 2.000 | Cohort ưu tiên đa dạng |
| --- | ---: | ---: | ---: |
| Trajectory | 1.390 | 4.000 | 4.000 |
| Task/instance khác nhau | 1.256 | 3.054 | **3.866** |
| Repo khác nhau | 656 | 1.133 | **1.501** |
| Trajectory vượt số task duy nhất | 134 | 946 | **134** |
| Tỷ trọng 10 repo nhiều nhất | 17,55% | 16,60% | **6,10%** |
| Token | 30.103.171 | 85.078.579 | 88.220.851 |

Phần bổ sung phủ **2.610 task mới** và **845 repo mới**. Trajectory bổ sung dài
nhất có **27.997 token**. Cần thêm **1.305 optimizer batch** từ mốc 695.
Tổng số token tăng khoảng 3,7% so với 4.000 trajectory đầu của lịch cũ;
không suy rằng thời gian train sẽ giảm chỉ nhờ chọn đa dạng hơn.

## Tạo và kiểm tra artifact

Chạy từ thư mục gốc repo, dùng Python environment của dự án:

```bash
.venv/bin/python -m scripts.data.prepare_diverse_continuation \
  --index data/processed/first_run_epoch_sft/index.json \
  --source-plan cloud/kaggle/recovery/continuation-plan.json \
  --resume-step 695 --target-step 2000 --verify-binaries \
  --output data/processed/first_run_diverse_4000
```

Thư mục dữ liệu đã chọn: `data/processed/first_run_diverse_4000/`.
Đây là manifest tham chiếu token/group binary trong
`data/processed/first_run_epoch_sft/`, không phải bản sao binary hay checkpoint.
Log, metrics và checkpoint của lần train vẫn nằm trong `outputs/training/first run/`.

Các file:

- `diversity-plan.json`: thứ tự source index, hash dữ liệu/lịch cũ/prefix và
  toàn bộ cohort, progress mong đợi, target batch và số liệu độ phủ.
- `trajectories.jsonl`: từng trajectory theo thứ tự, task/repo, độ dài, batch,
  rank và cờ `already_trained`.
- `summary.json`: metadata và độ phủ, không lặp lại mảng source index.
- `receipt.json`: SHA-256 của ba file, kết quả xác minh binary, trạng thái
  `checkpoint_verified=false` và `training_launched=false`.

Chạy lại cùng tham số cho cùng bytes. Chương trình từ chối ghi đè output chứa
lịch hoặc mức xác minh khác; dùng thư mục mới khi chủ động đổi lựa chọn.
`validate_plan` tính lại chính sách và đối chiếu toàn bộ manifest để phát hiện
thay đổi thứ tự, prefix, số liệu hoặc nguồn. Bộ chọn không tự loại trừ benchmark
lần nữa: nó chỉ dùng tập ứng viên của lịch nguồn đã được duyệt, không đưa thêm
sample từ corpus bên ngoài.

## Điều kiện trước khi chạy thật trên Vast

**Artifact là lịch dữ liệu đã chuẩn bị, chưa phải entrypoint có thể resume.**
Local chưa đọc được repo HF private khi chuẩn bị (HTTP 401), nên mốc 695 hiện
dựa trên tiến độ người dùng xác nhận. Trước khi train:

1. Xác thực HF bằng Secret/env riêng, lấy `latest.json` cùng manifest/payload
   từ một revision cố định và xác minh SHA-256 checkpoint.
2. Đối chiếu progress chính xác: `global_step=695`, `next_batch=695`,
   `samples_consumed=1390`, world size 2; đối chiếu identity và lịch continuation
   đã tạo ra prefix này. Nếu checkpoint mới hơn, tạo lại cohort giữ nguyên
   **toàn bộ prefix thực tế mới hơn**, không quay ngược cursor về 695.
3. Tích hợp view của cohort với launcher Vast và một chuyển đổi identity có
   kiểm soát từ **đúng checkpoint đã xác minh**. Giữ nguyên payload model,
   optimizer, scaler, RNG hai rank; không bỏ kiểm tra identity cũ hoặc giả rằng
   chỉ đổi `--index` là resume được. Ghi nhận rõ nếu thay CPU activation offload.
4. Khởi đầu ở batch 696, kết thúc tại batch 2.000. Kiểm tra save/resume và
   backup HF thật trước khi chạy không giám sát. Nếu còn dùng retention với
   history cleanup, phải giữ các ràng buộc repo/prefix chuyên dụng của publisher;
   không thêm tùy tiện một prefix khác vào cùng repo.

Chưa benchmark trên 3090 và chưa xác nhận đủ ngân sách 10 giờ/$5 để hoàn tất
1.305 batch còn lại. Thời gian cần đo bằng pilot với checkpoint resume thật.
