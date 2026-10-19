# Cấu hình QLoRA hiện tại

Cập nhật ngày 2026-10-18 theo yêu cầu train một epoch, tăng khả năng thích nghi
của LoRA và dùng trạng thái optimizer FP32. Áp dụng cho
`scripts/training/train_qlora_epoch.py`, `scripts/training/pilot_ddp_once.py` và
`scripts/training/pilot_optimized_qlora.py`. Phiên `first run` đã resume và publish
checkpoint 375 trên hai T4 với cấu hình này. Recovery 6 bị host RAM OOM trong
update 371. Recovery 8 chọn song song/lần lượt theo RAM nhưng lượt resume tiếp
theo báo CUDA OOM trong backward. Recovery 9 thay DDP wrapper bằng đồng bộ
gradient thủ công ở cả hai chế độ để bỏ bucket dư; đã hoàn tất 371–376 trên
CUDA nhưng OOM VRAM tại 377 với 29.991 token. Recovery 10 tiếp tục từ checkpoint
375, chỉ giữ các trajectory **chưa train có độ dài <28.000 token**, giữ nguyên
750 trajectory đã hoàn tất. Lịch hiệu lực: **9.966 trajectory, 210.967.717 token,
4.983 update**, gồm **4.608 update còn lại**. Không reset model/optimizer.
Chưa hoàn tất epoch hoặc đánh giá chất lượng; chưa chạy GPU với bộ lọc mới.

| Nhóm | Cấu hình |
| --- | --- |
| Base model | `Qwen/Qwen3.5-4B` |
| Revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Quantization | 4-bit NF4, double quantization |
| Compute / autocast | FP16 |
| LoRA | **r=32, alpha=64**, dropout=0.05, `alpha/r=2` |
| Tham số trainable | LoRA FP32; **64.929.792**, đã đo trên cả hai rank của `first run` version 8 |
| LoRA targets | Text-layer attention, GDN và MLP; danh sách bên dưới |
| Base weights | Frozen; tied embedding/lm_head FP16 |
| Phần nhạy cảm số học | Giữ các phép tính FP32 trong normalization/GDN của implementation |
| Optimizer | **bitsandbytes PagedAdamW32bit**, moment states FP32, giữ paging |
| Learning rate | `5e-5` |
| AdamW | `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01` |
| Gradient clipping | Max norm `1.0` |
| GradScaler | CUDA, initial scale `8.0` |
| Attention | `t4_repeated_kv`, memory-efficient SDPA |
| Gradient checkpointing | Nonreentrant, giữ dropout RNG |
| Saved activations | Offload sang CPU pinned memory |
| GDN block size | `1024`; mặc định optimized đã đồng bộ với DDP, có thể override bằng `--delta-block-size` |
| Loss chunk size | `64` target positions; vẫn giữ đầy đủ decoder context |
| KV cache | `use_cache=False` |
| Vision | Frozen, offload CPU, không dùng trong text SFT |
| Loss weights | Masked=0, reasoning=1, content=0.75, action=1.5, terminal call=2 |
| Scheduler | Không dùng; learning rate cố định |
| Lịch thực thi recovery 9 | Hai replica, không tạo DDP reducer/bucket; song song hoặc lần lượt theo RAM, all-reduce gradient SUM/2 trước unscale ở cả hai chế độ, một optimizer update/batch; dự phòng RAM tối thiểu 3 GiB/10% |
| Dữ liệu continuation recovery 10 | Giữ prefix 750 đã train; suffix còn 9.216 sample <28.000 token, bỏ 1.561 sample dài; giữ thứ tự tương đối seed 42, ghép lại batch 2, không cắt context |
| Checkpoint của epoch trainer | Sau update đầu tiên, mỗi 5 step, kết thúc/dừng có kiểm soát; giữ 1 bản local, tạm cần chỗ cho bản cũ + mới khi lưu |
| Checkpoint của các pilot | Mỗi 5 step, kết thúc/dừng có kiểm soát; mặc định giữ 2 bản local |
| Backup | HF private repo; epoch trainer hỗ trợ Model hoặc Dataset qua `--checkpoint-hub-repo-type`; chỉ bật khi có `--checkpoint-hub-repo` |

LoRA target modules trong các text decoder layer:

```text
q_proj, k_proj, v_proj, o_proj,
in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj,
gate_proj, up_proj, down_proj
```

## Chế độ hai GPU

- Hai T4, hai process; mỗi GPU giữ một text-model replica đầy đủ.
- Per-GPU batch size 1, global batch size 2, gradient accumulation 1.
- Trainer gốc và các pilot dùng DDP với `find_unused_parameters=False`,
  `broadcast_buffers=False`, `gradient_as_bucket_view=True`, bucket cap 25 MiB.
  Recovery 9 chặn constructor trước khi tạo reducer, broadcast adapter từ rank 0
  và tự trung bình gradient; NCCL P2P/IB vẫn tắt. Các field DDP trong config/hash
  gốc được giữ để kiểm tra checkpoint; backend thực tế ghi riêng trong
  `recovery-manifest.json` và event `manual_replica_initialized`.
- `train_qlora_epoch.py`: **một epoch** qua tất cả successful `train_pool`
  trajectory **dưới 30.000 token** từ toàn bộ success corpus. Không giới hạn
  vào 4.000 trajectory đã chọn. Giữ nguyên context, shuffle seed 42, không packing
  hoặc cắt ngắn; batch cuối lẻ dùng zero-loss padding với hệ số DDP tương ứng.
- `pilot_ddp_once.py` chỉ nhận đúng 2 sample thuộc hai task khác nhau và vẫn chỉ
  thực hiện **tổng cộng một optimizer step**.
- Ngưỡng 30k là điều kiện đầu vào, chưa bảo đảm cấu hình r=32/FP32 mới đủ VRAM
  tại độ dài đó. [Hướng dẫn epoch đầu tiên](first_run_epoch.md).
- Recovery 10 dùng view lọc trên chính index/binary gốc; không shuffle lại hoặc
  thay dataset nguồn. Ngưỡng **<28.000** chỉ áp dụng suffix chưa train tại
  checkpoint 375; 105 sample dài trong prefix đã train vẫn được giữ trong lịch.
  Lịch được cố định trong `continuation-plan.json` và identity checkpoint mới;
  resume sau này không tính lại ranh giới theo checkpoint latest.

Pilot optimized chạy batch size 1 trong một process. `single` đặt text model trên
GPU 0; `split` chia các layer qua hai GPU, không phải DDP. Mặc định thử cả hai
placement với hai step mỗi trajectory; script này không áp dụng ngưỡng 30k và
không cắt trajectory. Cần chọn input và placement chủ động trước khi khởi chạy.

## Khác biệt so với cấu hình đã đo

Các báo cáo GPU trước đây, gồm DDP job `20261016-150256-04d9cf`, dùng **r=16,
alpha=32, PagedAdamW8bit**. Những số đo thời gian, VRAM và độ dài chạy thành công
đó không phải kết quả của cấu hình mới. Script precision cũ
`pilot_weighted_sft.py` giữ cấu hình đối chiếu riêng, không được đổi theo bản này.

Tăng rank gấp đôi làm tăng cả adapter, gradient, optimizer states và DDP buffers;
chuyển moment states sang FP32 tăng thêm bộ nhớ. Với khoảng 64,93 triệu tham số,
riêng hai moment FP32 chiếm khoảng 0,48 GiB trước paging; đây không phải tổng VRAM.
Không giả định còn trống 3 GB/GPU hoặc cấu hình mới chắc chắn vừa T4 tại 30k token.
Tăng rank và độ chính xác optimizer cũng không tự bảo đảm chất lượng tốt hơn.

Log train r32/PagedAdamW32bit đã xác nhận trajectory **29.974 token** hoàn tất
update 72, peak allocated khoảng 13,82 GiB trên rank 1; batch 201 có hai sample
29.580/28.663 token và cũng hoàn tất. Vì vậy không được suy rằng cấu hình mới
chưa từng chạy gần 30k. Các mốc này không chứng minh mọi batch/resume vừa VRAM.
Recovery 8 serial giữ reducer DDP nhưng có thể cấp gradient riêng; một bộ
gradient FP32 cho 64.929.792 tham số là **247,69 MiB/GPU**. Recovery 9 loại bỏ
reducer ngay khi khởi tạo, không dùng MLP chunking, không giảm rank/context.
Backend này đã chạy thành công batch 371, nhưng serial vẫn OOM VRAM tại 377;
bỏ reducer không bảo đảm mọi sample gần 30k vừa T4. Bộ lọc 28k giảm yêu cầu bộ
nhớ, vẫn cần quan sát kết quả CUDA thực tế.

Đổi rank, optimizer và script làm đổi checkpoint identity. Không resume checkpoint
r=16/optimizer 8-bit vào cấu hình mới; không có bước tự chuyển đổi optimizer hoặc
adapter giữa hai cấu hình. Dùng thư mục checkpoint và Hub prefix mới cho lượt
train mới. Ngoại lệ có kiểm soát của recovery 10 chỉ chấp nhận **đúng manifest
checkpoint 375** đã kiểm SHA-256 để chuyển sang lịch suffix đã duyệt; vẫn kiểm
toàn bộ payload, config/package/math identity và giữ state/rank RNG. HF giữ cùng
prefix `first-run`, retention 3; publisher chỉ cho phép đúng parent này chuyển
sang đúng toàn bộ identity continuation. Không áp dụng cho thay rank/optimizer.
Xem [hướng dẫn checkpoint/resume](training_checkpoint_resume.md).

Phiên GPU version 5 lỗi trước khi đọc được Secret; version 8 do người dùng khởi
chạy qua giao diện đã đọc Secret thành công, load model và bắt đầu train.
Save/resume với PagedAdamW32bit trên CUDA và publish HF đã chạy thật đến
checkpoint 200; chưa kiểm chứng bit-exact CUDA, độ ổn định toàn epoch hoặc chất
lượng benchmark. Recovery 6 thêm dọn cache pinned trước save, watchdog OOM và
log tiến độ gọn, giữ nguyên cấu hình số học. Runtime packages vẫn theo
[weighted-requirements.txt](../cloud/kaggle/weighted-requirements.txt); giữ PyTorch
CUDA của Kaggle, không thay bằng bản CPU dùng cho test local.
