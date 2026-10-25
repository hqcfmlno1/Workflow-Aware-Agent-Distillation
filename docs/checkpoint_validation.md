# Đo lại validation loss cho các checkpoint FP16

So sánh base Qwen3.5-4B FP16, run token loss cũ và run group loss mới trên **cùng 20 trajectory holdout cố định**. Không train, không sửa adapter, không upload/xóa checkpoint. Mặc định tìm tất cả checkpoint còn tồn tại trên server và tại revision hiện tại của hai repo HF; ưu tiên bản local khi trùng. Bao gồm cả checkpoint local đã lưu xong nhưng upload HF thất bại.

Không thể tính loss ở các optimizer step không còn trọng số. Ví dụ lưu mỗi 10 step không thể khôi phục model ở step 11–19 từ checkpoint 10 và 20. CSV coverage ghi rõ khoảng trống này. Không dò lịch sử HF đã bị squash, không tự tạo số liệu.

## Chạy trên Vast sau khi pull

Dùng lại venv/token HF/cache của phiên train 5090. Dừng training trước khi đánh giá để tránh tranh GPU. Script không tự bật instance.

```bash
cd /workspace/Workflow-Aware-Agent-Distillation
git pull --ff-only origin sft/weighted-token-ce
bash cloud/vast/rtx5090/validate-checkpoints.sh --plan-only
```

Lệnh trên chỉ kiểm tra dữ liệu và liệt kê checkpoint, không load GPU. Nếu danh sách đúng, chạy tách khỏi SSH:

```bash
nohup bash cloud/vast/rtx5090/validate-checkpoints.sh > /workspace/validation-launch.log 2>&1 < /dev/null &
```

Xem log:

```bash
tail -f /workspace/Workflow-Aware-Agent-Distillation/outputs/validation/checkpoint-loss/validation.log
```

Chạy lại đúng lệnh sẽ tiếp tục từ trajectory chưa hoàn thành. Có khóa chống hai tiến trình cùng ghi output. Mỗi trajectory được flush/fsync vào JSONL; một lần ghi cuối bị ngắt sẽ được loại bỏ khi chạy tiếp. Dữ liệu/cache khác, adapter khác, mã đánh giá hoặc môi trường tính toán khác tạo fingerprint mới, không dùng nhầm kết quả trước.

Có thể chọn một số step để xem nhanh trước (base luôn được đo):

```bash
bash cloud/vast/rtx5090/validate-checkpoints.sh --steps 140 170 278 304
```

`--local-only` bỏ tìm checkpoint trên HF; bước chuẩn bị asset của wrapper vẫn có thể cần HF. Muốn kiểm tra hoàn toàn bằng asset local, gọi Python trực tiếp:

```bash
/workspace/venvs/sft5090/bin/python scripts/evaluation/validate_checkpoints.py --local-only --plan-only
```

Đổi danh sách run/path/revision bằng một bản sao của [config](../configs/evaluation/validation_fp16.json), truyền `--plan /path/plan.json`. `SFT5090_VENV`, `HF_HOME`, `CUDA_VISIBLE_DEVICES` và `VALIDATION_OUTPUT` có thể đặt trước lệnh shell. Không dùng launcher training để đo validation.

## Kết quả

Trong `outputs/validation/checkpoint-loss/`:

- `validation-loss.csv`: mỗi checkpoint một hàng, step/epoch, CE chung, CE từng nhóm, cả hai objective, delta so với base.
- `trajectory-loss.csv`: từng trajectory ở từng checkpoint, loss và số token của từng nhóm.
- `checkpoint-coverage.csv`: một hàng cho mỗi step tới mốc cuối đã biết; phân biệt đã đo, chờ đo, có weights nhưng không được chọn, và không có weights.
- `events.jsonl`: số token và tổng CE thô từng nhóm, thời gian, fingerprint; dùng để chạy tiếp.
- `evaluation.json`, `checkpoint-plan.json`: model/dataset/software và nguồn checkpoint để đối chiếu.
- `README.md`: bảng kết quả dễ đọc; `validation.log`: tiến độ trực tiếp.

Chỉ tải `manifest.json` và `adapter.pt` từ HF; **không tải `rank-*.pt` chứa optimizer**. Base chỉ load một lần. Từng adapter được kiểm SHA256, đúng tên/shape/dtype và hữu hạn trước khi chép vào model. Dataset/split phải khớp bundle đã chốt. Không đụng mã trainer đã được hash để bảo toàn khả năng resume.

## Cách đọc loss

Teacher forcing: model nhận toàn bộ lịch sử đúng của teacher, dự đoán token kế tiếp. Token observation/user vẫn ở trong context nhưng không vào loss. Không cắt trajectory. Vocabulary projection được chia block để tránh tạo tensor `sequence_length × vocabulary` quá lớn; đây không phải chunked prefill.

Gọi `CE_t` là cross entropy của token được giám sát, `N_g` là số token nhóm g và `S_g = sum CE_t` của nhóm đó. Bỏ vị trí đầu vì không có token trước để dự đoán.

- `ce_macro`: trung bình CE từng trajectory, mỗi trajectory một phiếu.
- `ce_micro`: tổng CE chia tổng số token được giám sát trên toàn holdout.
- `reasoning_ce`, `content_ce`, `action_ce`, `submit_ce`: CE micro trong từng nhóm, không nhân trọng số. Nhóm không có token để trống.
- `token_weighted_loss`: từng trajectory tính `sum_g w_g S_g / sum_g w_g N_g`, với `w = (1, .75, 1.5, 2)`, rồi lấy trung bình đều trên các trajectory. Tái hiện objective cũ.
- `group_weighted_loss`: từng trajectory tính `sum_g w_g N_g^(-.5) S_g / sum_g w_g N_g^(.5)`, với `w = (.15, .05, 1, .5)`, rồi lấy trung bình đều. Tái hiện objective mới.

**Cả hai objective đều được tính cho mọi checkpoint và base.** So cùng một cột giữa các checkpoint; không so token loss của run cũ với group loss của run mới. Delta âm nghĩa là loss thấp hơn base theo metric đó.

Loss giảm cho thấy dự đoán trajectory teacher tốt hơn; chưa chứng minh agent sửa được bug, tự submit tốt hơn hoặc ít loop khi rollout. Dùng `action_ce`/`submit_ce` cùng kết quả chạy task để đánh giá mục tiêu ưu tiên action.

## Kiểm chứng trước khi thuê GPU

```bash
.venv/bin/python -m pytest -q tests/test_validation_loss.py tests/test_fp16_training.py tests/test_fp16_group_loss.py
.venv/bin/python scripts/evaluation/validate_checkpoints.py --help
bash -n cloud/vast/rtx5090/validate-checkpoints.sh
```

Các test CPU kiểm tra causal alignment/masking bằng CE dense độc lập, hai công thức loss, macro/micro, integrity adapter, checkpoint discovery và phục hồi khi JSONL bị ghi dở. Việc chạy thực tế model 4B trên GPU cần được xác nhận trên server; test CPU không thay thế lần đo đó.
