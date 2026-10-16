# Lưu checkpoint định kỳ và tiếp tục train trên Kaggle

Hai script `scripts/training/pilot_optimized_qlora.py` và
`scripts/training/pilot_ddp_once.py` đã tích hợp checkpoint đầy đủ để tiếp tục
optimization. Mặc định lưu sau mỗi **5 optimizer step đã hoàn thành**, lưu khi
kết thúc bình thường hoặc dừng có kiểm soát, và giữ **2 checkpoint local** gần nhất.
Mỗi step ở đây đã gồm forward, loss, backward, cập nhật optimizer và GradScaler;
không lưu giữa backward hoặc giữa các microbatch.

Đây vẫn là các script **pilot**. `pilot_ddp_once.py` luôn chỉ có 2 sample,
1 sample/GPU, global batch size 2 và **tổng cộng đúng 1 optimizer step**. Nó lưu
checkpoint cuối ở step 1 dù chu kỳ mặc định là 5; resume checkpoint step 1 sẽ
thoát thành công mà không train lại. Chưa có trainer baseline một epoch hoàn chỉnh.
Không cần chạy lại pilot để bật tính năng trong code; các job cũ không thể được
khôi phục ngược thành checkpoint nếu khi đó chưa lưu trọng số và optimizer.

[Config hiện tại](qlora_current_config.md) đã chuyển sang LoRA r=32/alpha=64 và
PagedAdamW32bit. Checkpoint từ config cũ r=16/PagedAdamW8bit sẽ bị từ chối khi
resume sang config mới; cần dùng thư mục checkpoint/Hub prefix mới cho lượt train
mới. Chưa có cơ chế chuyển đổi checkpoint giữa hai config.

## Checkpoint chứa gì?

```text
checkpoints/
  latest.json
  checkpoint-00000005/
    manifest.json
    adapter.pt
    rank-00000.pt
    rank-00001.pt       # chỉ có khi DDP 2 rank
```

- `adapter.pt`: tất cả tham số trainable, hiện là LoRA FP32. Không lưu lại base
  model NF4 hoặc frozen embedding/head; chúng được dựng lại từ đúng model revision.
- `rank-*.pt`: optimizer, GradScaler, scheduler nếu có, RNG Python/NumPy/PyTorch
  CPU và CUDA của từng rank. Hai pilot hiện không dùng scheduler.
- `manifest.json`: global step, vị trí dữ liệu tiếp theo, identity của lượt train,
  số rank, danh sách file và SHA-256. Pilot optimized lưu cả record index và step
  trong record; DDP lưu vị trí batch 0 hoặc 1.
- `latest.json`: con trỏ tiện tra cứu. Resume `latest` vẫn quét và kiểm tra các thư
  mục hoàn chỉnh, bỏ qua bản ghi dở hoặc sai checksum để tìm bản hợp lệ gần nhất.

Ghi vào thư mục tạm trước, `fsync`, hoàn tất manifest rồi mới đổi tên để công bố
checkpoint. Mọi rank phải hoàn tất ghi và đồng ý vị trí dữ liệu. Chỉ rank 0 công bố
và xóa bản cũ. Giữ ít nhất 2 bản; thư mục tạm từ lần ghi lỗi có thể còn lại và bị bỏ
qua. Dọn bản cũ chỉ áp dụng cho checkpoint hợp lệ cùng identity trong thư mục đó.

Đây là định dạng training riêng của dự án. Không đưa `adapter.pt` trực tiếp vào
`PeftModel.from_pretrained()`. Thư mục `probe_only_adapter` của pilot optimized là
PEFT export riêng, không đủ để resume optimizer. Chỉ load checkpoint do chính bạn
tạo hoặc tin cậy: file trạng thái rank được đọc bằng Python pickle. SHA-256 kiểm
tra hỏng file, không xác thực người tạo.

## Tham số và đường dẫn

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--save-every-steps` | `5` | Số optimizer step hoàn thành giữa hai lần lưu |
| `--keep-checkpoints` | `2` | Số bản local giữ lại; tối thiểu 2 |
| `--checkpoint-dir` | Theo script | Thư mục gốc chứa checkpoint |
| `--resume-from` | Không có | Đường dẫn một thư mục `checkpoint-*`, hoặc `latest` |
| `--checkpoint-hub-repo` | Không có | Repo HF **model private có sẵn**, dạng `owner/name` |
| `--checkpoint-hub-prefix` | `training` | Namespace trên repo; script tự thêm placement/experiment |
| `--max-train-seconds` | `900` | Ngân sách giây mỗi worker, bắt đầu trước khi load model |
| `--stop-margin-seconds` | `120` | Dự phòng lưu/upload trước khi hết ngân sách |
| `--expected-step-seconds` | `180` | Ước lượng step tiếp theo trước khi bắt đầu |

Pilot optimized mặc định lưu tại
`outputs/training/optimized_qlora/{single,split}/checkpoints/`. Nếu truyền
`--checkpoint-dir outputs/training/recovery`, nó lưu tại
`outputs/training/recovery/single/` hoặc `.../split/`. `split` là model partition
trong một process, không phải DDP. Khi resume chỉ chọn đúng placement của bản lưu.

DDP mặc định lưu tại `outputs/training/ddp_once/checkpoints/`; nếu truyền
`--checkpoint-dir`, DDP dùng đúng thư mục đó, không thêm hậu tố.

Log stdout có dòng JSON `event: checkpoint_saved`, gồm `path`, `global_step` và
`published`. Dùng `krun logs JOB_ID` để theo dõi; metadata/report nằm trong output
của script. `published: true` chỉ được in sau khi API Hub trả về thành công.
Khi resume có cấu hình Hub, dòng `checkpoint_republished` xác nhận bản đã load
được gửi lại hoặc đã tồn tại nguyên vẹn trên remote.

## Hết quota và dừng an toàn

Trước mỗi step, worker kiểm tra thời gian đã dùng + thời gian dự kiến của step
tiếp theo + khoảng dự phòng. Nếu vượt ngân sách, nó lưu ở ranh giới step rồi trả
trạng thái `stopped_checkpointed`. Pilot optimized cập nhật ước lượng bằng thời
gian step vừa đo, lấy ít nhất giá trị cấu hình; độ dài trajectory có thể làm step
tiếp theo lâu hơn dự kiến. Với DDP, yêu cầu dừng từ một rank dừng cả hai rank.

`SIGINT`/`SIGTERM` chỉ đặt cờ; worker hoàn thành step đang chạy rồi mới lưu. Không
có lưu khẩn cấp được bảo đảm khi OOM, `SIGKILL`, mất máy hoặc Kaggle cắt phiên
đột ngột. Có thể mất phần công việc sau checkpoint gần nhất. Chu kỳ 5 tương ứng
thường dưới 5 update khi bị ngắt giữa các lần lưu; chưa có checkpoint nào trước
lần lưu đầu tiên. Có thể dùng chu kỳ 1 khi ưu tiên giảm mất tiến độ, đổi lại tốn
thời gian và dung lượng ghi/upload hơn.

`--max-train-seconds` là **ngân sách tự đặt**, không đọc quota còn lại từ Kaggle.
Nó tính cả load model trong worker, nhưng không tính phần cài dependency và chuẩn
bị/download ở launcher. `--cell-timeout` của KRun phải dành thêm thời gian cho các
phần đó, dừng worker, ghi và upload checkpoint. Khoảng dự phòng 120 giây không
đảm bảo đủ với mọi tốc độ mạng; nên chừa nhiều phút theo kích thước checkpoint
và thời gian upload thực tế. Đóng cửa sổ xem log không dừng GPU remote.
Launcher có hard timeout sau ngân sách worker thêm 600 giây với optimized hoặc
300 giây với DDP. Khi chạm mốc này, nó kết thúc cả nhóm process, gồm các rank con
của torchrun; chỉ những checkpoint đã hoàn tất trước đó dùng được.

**Checkpoint chỉ ở `/kaggle/working` chưa bảo vệ khỏi mất cả phiên.** Cấu hình
backup ngoài phiên, hoặc tải artifact về khi phiên còn hoạt động. `--output` của
KRun khai báo artifact cần lấy; nó không phải cơ chế upload mỗi checkpoint.

## Backup định kỳ lên Hugging Face private

Tạo trước một repo **model private** và một token có quyền ghi vào repo. Thêm
`HF_TOKEN` vào Kaggle Secrets, cho phép notebook sử dụng secret, rồi truyền tên
secret bằng `--secret HF_TOKEN` của KRun. Không ghi giá trị token vào lệnh, docs,
file nguồn hay log. Internet phải được bật cho upload/download.

Thêm các tùy chọn sau vào lệnh KRun **đã định chạy**; đây chỉ là mẫu cấu hình,
không phải yêu cầu chạy thêm pilot:

```text
# Trước dấu -- của KRun:
--secret HF_TOKEN
--output outputs/training/recovery

# Sau dấu --, truyền cho pilot:
--checkpoint-dir outputs/training/recovery
--save-every-steps 5
--keep-checkpoints 2
--checkpoint-hub-repo YOUR_USER/YOUR_PRIVATE_MODEL_REPO
--checkpoint-hub-prefix qlora-run-20261016
--stop-margin-seconds 300
--expected-step-seconds 240
```

Prefix thực tế sẽ là `qlora-run-20261016/single`, `.../split` hoặc `.../ddp_once`.
Mỗi thí nghiệm mới dùng một prefix mới. Tiếp tục cùng lượt train mới dùng lại
prefix cũ. Script kiểm tra repo private trước khi train; mỗi lần lưu đăng toàn bộ
file của checkpoint và cập nhật con trỏ trong **một Hub commit**. So identity và
global step với bản remote hiện có, từ chối thí nghiệm khác hoặc ghi lùi/ngang step
với nội dung khác. Gửi lại đúng bản có cùng manifest là thao tác an toàn;
commit cũng kiểm tra parent revision để tránh hai process ghi đè đồng thời.

Nếu upload lỗi, script báo lỗi/dừng, giữ bản local mới và các bản local trước đó,
không tiếp tục train âm thầm. Remote vẫn có commit hoàn chỉnh gần nhất nếu đã từng
upload thành công. Resume từ bản remote đó, hoặc từ bản local mới nếu đã lấy được.
Khi resume cùng cấu hình Hub, script gửi lại bản đã load trước khi train tiếp,
kể cả checkpoint DDP step 1 đã hoàn thành; không chạy lại optimizer step để gửi
lại checkpoint.
Nếu API đã commit nhưng mất phản hồi mạng, kiểm tra/tải lại bản remote mới nhất.
Không thể đảm bảo backup nếu mạng hoặc quyền ghi lỗi trước lần upload đầu tiên.

Chính sách giữ 2 bản **chỉ áp dụng local**. Các checkpoint và lịch sử commit trên
Hub chưa được tự xóa, nên dung lượng repo tăng theo số lần lưu. Cần lập ngân sách
lưu trữ riêng; không đưa token hoặc dữ liệu nhạy cảm vào repo công khai.

## Resume trong phiên mới

1. Tải checkpoint hoàn chỉnh từ repo private trên máy có token đọc repo:

   ```bash
   .venv/bin/python scripts/training/fetch_training_checkpoint.py \
     --repo YOUR_USER/YOUR_PRIVATE_MODEL_REPO \
     --prefix qlora-run-20261016/single \
     --output outputs/training/restored
   ```

   Công cụ không load model hay dùng GPU. Nó chốt một Hub revision cho cả con trỏ
   lẫn file, kiểm tra SHA-256 và in ra đường dẫn, ví dụ
   `outputs/training/restored/qlora-run-20261016/single/checkpoint-00000005`.
   DDP dùng prefix kết thúc bằng `/ddp_once`. Cần `huggingface_hub` trong môi trường.

2. Khi khởi chạy phiên Kaggle mới, gắn checkpoint bằng `--input DIR` trước dấu
   `--` của KRun (giống cách gắn JSON records). Sau dấu `--`, thêm:

   ```text
   --resume-from outputs/training/restored/qlora-run-20261016/single/checkpoint-00000005
   --checkpoint-dir outputs/training/recovery
   --placements single
   ```

   Dòng `--placements single` chỉ dành cho pilot optimized. Với DDP, truyền đủ
   checkpoint của cả hai rank và chạy bằng launcher DDP vốn có trong script.
   Mẫu đường dẫn dùng project-relative để KRun ánh xạ đúng; không dùng đường dẫn
   tuyệt đối của máy local làm đường dẫn remote.

3. Nếu checkpoint đã nằm ở đúng thư mục mặc định/cấu hình trong phiên hiện tại,
   có thể dùng `--resume-from latest`. Khi restore vào thư mục khác như ví dụ
   trên, dùng đường dẫn cụ thể. Các checkpoint tiếp theo ghi vào checkpoint-dir
   mới. Đọc metadata để xác nhận step tiếp tục; script không lặp lại update đã lưu.

Giữ **nguyên file JSON records theo byte và thứ tự**, model revision, code tính
toán/script, các package được ghi trong manifest, config training, placement và
số rank. Đổi dữ liệu, learning rate hoặc world size không được coi là resume hợp
lệ. Có thể thay chu kỳ lưu, số bản giữ, nơi lưu/Hub prefix và ngân sách thời gian.
Base weights phải có lại trong cache hoặc tải lại đúng revision. Không có cơ chế
chuyển checkpoint từ DDP 2 rank sang single GPU hay ngược lại.

## Đã kiểm tra và giới hạn hiện tại

Test CPU dùng model nhỏ với dropout, frozen base và adapter trainable: resume
khớp chính xác loss, tham số, optimizer, scheduler và scaler của lượt chạy liên
tục. Test Gloo dùng hai process DDP thật kiểm tra RNG từng rank, resume đồng bộ
và lan truyền lỗi khi một rank không thể lưu. Test khác kiểm tra bản hỏng, ghi dở,
sai identity, giữ bản trước khi upload lỗi, và commit/download Hub bằng transport
mô phỏng. CLI `--help` không load model hoặc khởi chạy train.

Chưa chạy kiểm chứng save/resume trên CUDA với PagedAdamW32bit hoặc upload thật lên
Hub; kết quả CPU không chứng minh bit-exact trên T4/bitsandbytes. Tính năng này
được bổ sung mà không gửi thêm job Kaggle/GPU. Các kết quả pilot GPU trước đây
trong [Kaggle README](../cloud/kaggle/README.md) vẫn chỉ chứng minh những lần chạy
đã đo, chưa chứng minh long-run training hoặc full-state resume trên GPU.
