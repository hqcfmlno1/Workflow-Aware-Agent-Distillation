# Cấu hình QLoRA hiện tại

Cập nhật ngày 2026-10-17 theo yêu cầu train một epoch, tăng khả năng thích nghi
của LoRA và dùng trạng thái optimizer FP32. Áp dụng cho
`scripts/training/train_qlora_epoch.py`, `scripts/training/pilot_ddp_once.py` và
`scripts/training/pilot_optimized_qlora.py`. Phiên `first run` đã resume và publish
checkpoint 200 trên hai T4 với cấu hình này. Version 15 hoàn tất update 205 nhưng
host RAM OOM khi lưu checkpoint. Chưa hoàn tất epoch hoặc đánh giá chất lượng.

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
| Checkpoint của epoch trainer | Sau update đầu tiên, mỗi 5 step, kết thúc/dừng có kiểm soát; giữ 1 bản local, tạm cần chỗ cho bản cũ + mới khi lưu |
| Checkpoint của các pilot | Mỗi 5 step, kết thúc/dừng có kiểm soát; mặc định giữ 2 bản local |
| Backup | HF private repo; epoch trainer hỗ trợ Model hoặc Dataset qua `--checkpoint-hub-repo-type`; chỉ bật khi có `--checkpoint-hub-repo` |

LoRA target modules trong các text decoder layer:

```text
q_proj, k_proj, v_proj, o_proj,
in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj,
gate_proj, up_proj, down_proj
```

## Chế độ DDP hiện có

- Hai T4, hai process; mỗi GPU giữ một text-model replica đầy đủ.
- Per-GPU batch size 1, global batch size 2, gradient accumulation 1.
- `find_unused_parameters=False`, `broadcast_buffers=False`,
  `gradient_as_bucket_view=True`, bucket cap 25 MiB; NCCL P2P/IB tắt.
- `train_qlora_epoch.py`: **một epoch** qua tất cả successful `train_pool`
  trajectory **dưới 30.000 token** từ toàn bộ success corpus. Không giới hạn
  vào 4.000 trajectory đã chọn. Giữ nguyên context, shuffle seed 42, không packing
  hoặc cắt ngắn; batch cuối lẻ dùng zero-loss padding với hệ số DDP tương ứng.
- `pilot_ddp_once.py` chỉ nhận đúng 2 sample thuộc hai task khác nhau và vẫn chỉ
  thực hiện **tổng cộng một optimizer step**.
- Ngưỡng 30k là điều kiện đầu vào, chưa bảo đảm cấu hình r=32/FP32 mới đủ VRAM
  tại độ dài đó. [Hướng dẫn epoch đầu tiên](first_run_epoch.md).

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

Đổi rank, optimizer và script làm đổi checkpoint identity. Không resume checkpoint
r=16/optimizer 8-bit vào cấu hình mới; không có bước tự chuyển đổi optimizer hoặc
adapter giữa hai cấu hình. Dùng thư mục checkpoint và Hub prefix mới cho lượt
train mới. Xem [hướng dẫn checkpoint/resume](training_checkpoint_resume.md).

Phiên GPU version 5 lỗi trước khi đọc được Secret; version 8 do người dùng khởi
chạy qua giao diện đã đọc Secret thành công, load model và bắt đầu train.
Save/resume với PagedAdamW32bit trên CUDA và publish HF đã chạy thật đến
checkpoint 200; chưa kiểm chứng bit-exact CUDA, độ ổn định toàn epoch hoặc chất
lượng benchmark. Recovery 6 thêm dọn cache pinned trước save, watchdog OOM và
log tiến độ gọn, giữ nguyên cấu hình số học. Runtime packages vẫn theo
[weighted-requirements.txt](../cloud/kaggle/weighted-requirements.txt); giữ PyTorch
CUDA của Kaggle, không thay bằng bản CPU dùng cho test local.
