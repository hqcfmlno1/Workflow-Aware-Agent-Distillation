# Cấu hình QLoRA hiện tại

Cập nhật ngày 2026-10-16 theo yêu cầu tăng khả năng thích nghi của LoRA và dùng
trạng thái optimizer FP32. Áp dụng cho `scripts/training/pilot_ddp_once.py` và
`scripts/training/pilot_optimized_qlora.py`. Đây là cấu hình code hiện tại, chưa
được chạy GPU để xác nhận bộ nhớ hoặc chất lượng sau thay đổi.

| Nhóm | Cấu hình |
| --- | --- |
| Base model | `Qwen/Qwen3.5-4B` |
| Revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Quantization | 4-bit NF4, double quantization |
| Compute / autocast | FP16 |
| LoRA | **r=32, alpha=64**, dropout=0.05, `alpha/r=2` |
| Tham số trainable | LoRA FP32; khoảng 64,93 triệu tham số, suy ra từ cấu hình r=16 đã đo |
| LoRA targets | Text-layer attention, GDN và MLP; danh sách bên dưới |
| Base weights | Frozen; tied embedding/lm_head FP16 |
| Phần nhạy cảm số học | Giữ các phép tính FP32 trong normalization/GDN của implementation |
| Optimizer | **bitsandbytes PagedAdamW32bit**, moment states FP32, giữ paging |
| Learning rate | `5e-5` |
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
| Scheduler | Chưa có trong các pilot hiện tại |
| Checkpoint | Mỗi 5 step; thêm khi kết thúc/dừng có kiểm soát; giữ 2 bản local |
| Backup | HF model repo private, chỉ bật khi cấu hình `--checkpoint-hub-repo` |

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
- Chỉ nhận đúng 2 successful train-pool sample thuộc hai task khác nhau, mỗi
  trajectory **dưới 30.000 token**, giữ nguyên toàn bộ history.
- `pilot_ddp_once.py` vẫn chỉ thực hiện **tổng cộng một optimizer step**. Chưa có
  trainer DDP baseline một epoch. Ngưỡng 30k là điều kiện đầu vào, chưa bảo đảm
  cấu hình mới đủ VRAM tại độ dài đó.

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

Cập nhật này chỉ sửa code/docs và kiểm tra CPU, không gửi thêm job Kaggle/GPU.
Save/resume với PagedAdamW32bit trên CUDA và chất lượng benchmark chưa được kiểm
chứng. Runtime packages vẫn theo
[weighted-requirements.txt](../cloud/kaggle/weighted-requirements.txt); giữ PyTorch
CUDA của Kaggle, không thay bằng bản CPU dùng cho test local.
