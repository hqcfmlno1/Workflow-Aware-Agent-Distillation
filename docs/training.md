# Review và chạy QLoRA

Luồng refactor tại local bắt đầu ở [`scripts/train.py`](../scripts/train.py),
đi vào [`training/cli.py`](../src/workflow_distillation/training/cli.py), rồi
[`training/engine.py`](../src/workflow_distillation/training/engine.py).
Code này chưa được triển khai lên server và chưa pilot QLoRA trên CUDA.
Các launcher mà phiên remote hiện có đã dùng vẫn được giữ nguyên.

## Bản đồ đọc code

Đọc theo thứ tự sau; đường dẫn module tính từ `src/workflow_distillation/`.

| Thứ tự | File | Nội dung cần review |
| --- | --- | --- |
| 1 | `training/config.py` | Model/revision, objective, LoRA, optimizer; schema GPU profile |
| 2 | `training/data.py` | Xác minh pool, thứ tự trajectory, batch transition và batch cuối |
| 3 | `training/cli.py` | Kiểm tra đầu vào, preflight, launch hai worker |
| 4 | `training/model.py` | NF4 base, LoRA target, FP32 adapter, checkpointing, optimizer |
| 5 | `training/activation_storage.py`, `training/memory.py` | Saved tensors CPU/hybrid, RAM admission và serial execution |
| 6 | `training/loss.py`, `weighted_sft_loss.py`, `weighted_sft_batches.py` | Weighted-token CE, trung bình đều theo trajectory, chia slot cho rank |
| 7 | `training/engine.py`, `training/distributed.py`, `training/device.py` | Forward/backward, scaled gradient mean, unscale, clip, update |
| 8 | `training/identity.py`, `training/compatibility.py`, `training/checkpoints.py` | Identity mới, ranh giới migration, strict resume và publish HF |
| 9 | `training/events.py`, `training/monitor.py`, `training/process.py` | JSON events, tiến độ, watchdog và subprocess thuộc launcher |

Các phép tính dùng chung vẫn nằm ở `weighted_sft_loss.py`, `t4_attention.py`,
`optimized_qlora.py`; đọc/ghi checkpoint atomic dùng `training_checkpoints.py`.
Luồng mới gọi trực tiếp các primitive này, không chạy chuỗi launcher cũ.

## Cách review một thay đổi

1. Đọc diff và xác định nó đổi objective, dữ liệu, execution hay checkpoint.
   Bắt đầu ở entrypoint mới, chỉ đi vào launcher cũ khi review compatibility.
2. Với loss/batch: kiểm label shift, mask/padding, denominator mỗi trajectory,
   coefficient batch cuối; đối chiếu gradient/update với single-process reference.
3. Với model/memory: kiểm frozen/trainable, dtype/device, LoRA targets,
   recompute RNG, giới hạn GPU storage và RAM admission trước forward.
4. Với distributed: kiểm mọi rank đi qua cùng collective, scaled gradient mean
   trước unscale/clip; serial vẫn phải cho cùng objective với parallel.
5. Với resume: kiểm hash/package/profile/order/cursor; giữ optimizer/scaler/RNG;
   save ở optimizer boundary, commit local trước HF, không rollback remote latest.
6. Chạy test phù hợp rồi toàn bộ suite. Nếu đổi CUDA path, thêm pilot cùng
   checkpoint/batch khó trên GPU trước deploy; test CPU không xác nhận peak VRAM.

| Thay đổi | Test cần đọc trước |
| --- | --- |
| GPU profile, pool, lịch | `tests/test_training_config_data.py` |
| Loss, gradient, RAM policy | `tests/test_training_numerics.py` |
| Step thật trên hai rank, batch cuối, serial | `tests/test_training_engine_distributed.py` |
| Migration, state/RNG và HF atomicity | `tests/test_training_migration.py` |
| Preflight, child identity/source drift, stop boundary | `tests/test_training_cli_engine.py` |
| Log, OOM watchdog, process ownership/timeout | `tests/test_training_monitor.py` |

Khi ghi finding, nêu **file/dòng → trigger → tác động → bằng chứng → cách kiểm**.
Không sửa inventory hash chỉ để test qua: code mới cần identity và migration
phù hợp. Ghi rõ kết quả nào đo trên CPU, kết quả nào đã đo trên GPU.

## Cấu hình theo GPU

Mỗi loại GPU có một thư mục dưới [`configs/training/gpus/`](../configs/training/gpus/).
`profile.json` là cấu hình đầy đủ cho entrypoint mới. Model và loss dùng chung;
không sao chép toàn bộ trainer cho mỗi GPU.

| Profile | Số GPU | Trajectory/GPU | Global batch | Saved activations |
| --- | ---: | ---: | ---: | --- |
| `t4_16gb` | 2 | 1 | 2 | Pinned CPU; serial nếu policy RAM yêu cầu |
| `rtx3090_24gb` | 2 | 2 | 4 | Pinned CPU |
| `rtx4080_32gb` | 2 | 2 | 4 | Giữ tối đa 4 GiB/GPU; phần còn lại pinned CPU |

Profile kiểm tra tên và VRAM thực tế. `rtx4080_32gb` dành riêng cho biến thể
đã đo có 32 GiB; RTX 4080 thông thường 16 GiB không đạt admission.
Attention giữ repeated-KV Efficient SDPA. Flash và full-GPU activation storage
không được schema production mới chấp nhận. Kết quả pilot cũ ở
[README 4080](../configs/training/gpus/rtx4080_32gb/README.md) chỉ là bằng chứng
cho implementation trước refactor.

## Những bất biến được giữ

- Qwen3.5-4B/revision cũ; NF4 double quantization, FP16 compute, FP32 LoRA
  `r=32`, `alpha=64`, dropout `0.05`; PagedAdamW32bit và optimizer settings cũ.
- Trọng số nhóm token: masked `0`, reasoning `1`, content `0.75`, action `1.5`,
  submit `2`. Mỗi trajectory được chuẩn hóa bằng tổng trọng số token được học;
  global loss là trung bình đều các trajectory thật. Batch cuối thiếu sample
  có coefficient riêng để bản sao lấp rank không được đếm thêm.
- Pool/order/prefix, batch-size transition đã ghi trong checkpoint, cursor
  và số trajectory đã tiêu thụ được đối chiếu trước khi train. Suffix chưa học
  phải chứa trajectory nguyên vẹn dưới 28.000 token.
- Hai replica đầy đủ, không model sharding. Broadcast adapter lúc khởi tạo;
  SUM/world-size các scaled gradient trước unscale/clip/update. Không tạo DDP
  reducer/bucket. Serial chỉ thay lịch forward/backward giữa hai rank.
- Model, optimizer, scaler, RNG mỗi rank và cursor được restore từ checkpoint.
  Không có cơ chế bỏ qua hash hoặc tự fallback sang train lại.

Không thay global `DistributedDataParallel`, `save_on_cpu`, `Popen` hay các
method optimizer để điều khiển training. Transformers vẫn cần đăng ký attention
và kernel delta-rule khi build model; chúng nằm rõ trong `training/model.py`.

## Preflight và migration

Entrypoint mới hiện hỗ trợ continuation của pool đã mở rộng, cùng global batch
với parent. Profile T4 mô tả execution và có regression test policy; nó không
tự chuyển checkpoint Kaggle cũ thành pool hoặc đổi batch 4 về batch 2.

Chuẩn bị đúng environment/dependencies của checkpoint, model snapshot trong
HF cache và checkpoint đầy đủ tại local. Không đưa token vào command hoặc Git.
Xem cú pháp bằng `python scripts/train.py --help`.

Ví dụ CPU preflight cho migration từ một checkpoint legacy của pool hiện có
(thay `PARENT` bằng đường dẫn checkpoint thực tế):

```bash
PARENT=/absolute/path/to/checkpoint-00001028
PARENT_SHA=$(sha256sum "$PARENT/manifest.json" | cut -d ' ' -f 1)

.venv/bin/python scripts/train.py \
  --index data/processed/first_run_epoch_sft/index.json \
  --source-plan cloud/kaggle/recovery/continuation-plan.json \
  --cohort data/processed/first_run_diverse_4000/diversity-plan.json \
  --pool-plan data/processed/first_run_diverse_remaining/pool-plan.json \
  --gpu-profile configs/training/gpus/rtx4080_32gb/profile.json \
  --resume-from "$PARENT" \
  --migration-parent-manifest "$PARENT/manifest.json" \
  --migration-parent-manifest-sha256 "$PARENT_SHA" \
  --output outputs/training/first_run/refactor-preflight \
  --preflight-only
```

Preflight xác minh binary/hash, plan, packages, parent và cursor; không allocate
model CUDA, không publish HF. `cuda_pilot_verified: false` trong receipt nghĩa là
chưa chứng minh forward/backward trên GPU. Không dùng receipt này thay pilot.

Trước khi dùng luồng mới cho run dài, trên máy có GPU cần pilot từ cùng parent
và các batch khó đã dùng trước đây: so loss/gradient/update, peak VRAM/RAM,
save/reload và publish/download checkpoint. Dùng output mới; khi chạy thật
bỏ `--preflight-only`, thêm `--stop-at-step` cho pilot và các tham số HF nếu cần
kiểm tra publication. Mặc định budget là 32.400 giây, không phải chạy vô hạn.

Migration lần đầu bắt buộc pin SHA256 manifest parent. Child ghi identity mới
và lineage đầy đủ; không giả nhận bitwise equivalence với code cũ. Resume child
dùng `--resume-from` trỏ tới child, bỏ hai cờ migration. Code/profile/package/
data thay đổi sẽ bị từ chối nếu không có migration phù hợp.

## Log và dừng ở checkpoint

Mỗi lần launch dùng thư mục output mới để log lỗi cũ không kích hoạt watchdog.
Trong output có:

- `preflight.json`, `launch.json`, `hardware-audit.json`: đầu vào/identity/hardware.
- `rank0.jsonl`, `rank1.jsonl`: event chi tiết; `step` chỉ ghi sau optimizer update.
- `first run.log`: stdout/stderr torchrun; tee có thể lặp dòng hiển thị.
- `telemetry/progress.log`, `telemetry/status.json`: step, trajectory, loss,
  checkpoint hoàn tất và phase mỗi rank; reader chỉ đọc phần đuôi log có giới hạn.
- `checkpoints/`: mặc định checkpoint local; `checkpoint_complete` xảy ra sau
  save và publisher đã trả về. Nếu HF publish fail, checkpoint local đã commit
  vẫn còn, run báo lỗi và có thể republish khi resume.

```bash
tail -n 20 -F outputs/training/first_run/NAME/telemetry/progress.log
touch outputs/training/first_run/NAME/STOP
```

`STOP` yêu cầu dừng tại ranh giới optimizer rồi save. Watchdog khi rank fail hoặc
cgroup OOM dọn cây process torchrun đã đăng ký; không bảo đảm cứu batch đang
OOM. Nó không tự Stop/Destroy Vast instance hoặc quản lý billing.

## Tương thích và kiểm thử

Xem [`legacy/`](../legacy/README.md) cho source bị checkpoint pin theo path/hash.
Những file đó được giữ nguyên byte tại đường dẫn cũ; entrypoint mới không chạy
bootstrap Kaggle hoặc chain import launcher legacy. `.krunignore` đã được gỡ;
`.krun/` vẫn ignore artifact KRun cũ.

Test mới có profile/data/hash guards, loss và gradient batch đủ/thiếu,
hai process Gloo chạy Step thật cả parallel/serial, restore optimizer/scaler/RNG,
atomic HF migration/retention/rollback và watchdog ownership.

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/workflow_distillation/training scripts/train.py tests/test_training_*.py
git diff --check
```

## Fresh FP16 LoRA on one RTX5090

See the [RTX5090 runbook](../configs/training/gpus/rtx5090_32gb/README.md). This
independent experiment starts a new adapter from the pinned FP16 base, uses the
TQS-selected corpus with a fixed20task holdout, and retains epoch checkpoints.
Its CPU preflight and pilot are separate from the two-GPU NF4 migration above.
