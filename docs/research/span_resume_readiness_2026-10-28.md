# Kiểm tra trước khi resume scoring và training

Ngày 2026-10-28; branch `sft/offline-span-anchor`. Instance đã dừng theo yêu cầu;
kiểm tra này chỉ dùng CPU local, không bật server hay dùng GPU.

## Cache thực tế đã được kiểm tra

Bản sao lưu SHA256 `220748dc70ff69f1b057c6eab5fda54626f7d028ab29088355f03bf138067470`
được giải nén riêng, không sửa cache gốc. Đối chiếu toàn bộ phần đã lưu:

- 1.060/1.180 trajectory hoàn tất; còn 120 trajectory chưa hoàn tất.
- 27.931 eligible turn đã có weight; 6.933.421 supervised target token trong
  các trajectory hoàn tất.
- Record 1060 (đánh số từ 0), message index 20 còn 3 variant logps đã lưu.
- Corpus/provenance hash và mọi scoring-source hash khớp identity của cache.
- Reference logps và anchor weights khớp đúng vị trí target token.
- Tính lại delta/rank/span weights, rồi map overlap xuống token: sai lệch 0 sau
  float32 conversion. Non-reasoning CE weights giữ nguyên 1.
- Chưa có final index là đúng trạng thái partial; chưa được dùng để train.

Số liệu đầy đủ: [backup audit](span_resume_backup_audit_2026-10-28.json).
Việc này kiểm tra nội dung đã lưu, không chứng minh mọi GPU forward trước đó đúng
hay scorer cache tương đương số học với full prefill. Khác biệt BF16 đã được chấp
thuận và ghi trong [profile](../../configs/training/gpus/rtx5090_32gb/METACURATE_SPAN_ANCHOR.md).

## Sửa đường triển khai

Launcher mặc định chuyển sang config/cache `*_cached` đang dùng, tránh vô tình
chạy scorer full cũ. Thêm `--stage all --foreground` để supervisor giữ quyền quản
lý cả scoring và training. Chỉ vào train khi sidecar đầy đủ, identity/checksum và
alignment qua preflight. Không đổi scoring source, công thức loss hay config số.
Pin thêm Triton 3.6.0 và Tokenizers 0.23.3 theo đúng runtime đã score.

```bash
export SFT5090_VENV=/venv/main
export HF_HOME=/workspace/.hf_home
bash scripts/training/run_metacurate_span_anchor.sh --stage all --foreground --max-seconds 28800
```

Chạy qua supervisor với `stopasgroup=true`, `killasgroup=true`, `stopwaitsecs=600`,
`autostart=false`, `autorestart=false`. Khi scoring xong, script kiểm tra đúng
một trajectory dài nhất nếu chưa có pilot hợp lệ rồi khởi tạo LoRA mới để train;
không tiếp tục adapter cũ. Restart sau khi đã train thì `all` dùng local latest.
Resume từ máy khác bằng checkpoint HF cần `--resume hub` rõ ràng.

Training giữ BF16 base không quantize, FP32 LoRA r32/alpha64, accumulation4,
LR1e-4/cosine/warmup0.05, 2 epoch trên đủ1.180 trajectory, không holdout.
Local recovery mỗi20 optimizer update, giữ bản mới nhất và endpoint295/590.
HF chỉ upload cuối epoch; log có total/weighted/plain CE, anchor và NLL theo field.
Không có validation loss vì không tách holdout.

## Phạm vi kiểm tra và điều kiện còn lại

Focused CPU tests bao phủ scoring resume, cache integrity, loss/gradient so với
công thức dense, accumulation, checkpoint retention, failed upload và event-log
recovery. Shell tests xác nhận routing, partial/failed score không vào train,
foreground execution và local auto-resume; chúng không giả lập GPU thật hay HF.
Một reviewer độc lập không tìm thấy lỗi critical/important trong thay đổi launcher.

Trước khi chạy trên server: cập nhật code/wrapper, khớp exact runtime của cache,
kiểm tra HF write credential/private destination và dung lượng trống. GPU admission
của objective mới vẫn cần thực hiện trên server, đúng một trajectory dài nhất.
Đây là kết luận sẵn sàng về code/cache local, không phải xác nhận training đã chạy.
