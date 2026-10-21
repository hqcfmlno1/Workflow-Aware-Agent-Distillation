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

## Xác minh checkpoint và chuyển sang Vast

Ngày 2026-10-19 đã xác thực repo HF private và kiểm tra đầy đủ payload của
checkpoint 695 trên Vast, tại revision `729b26d96172a93218defba540e8d11ae23d8b97`.
Manifest SHA-256: `2cdd8808a886c3f32ac887b3411ad8dae3d91ccb8363a6206c48bf8c735c2b95`.
`receipt.json` của bước chọn dữ liệu vẫn ghi trạng thái tại lúc tạo artifact;
kết quả xác minh remote nằm trong `outputs/training/first run/vast-setup/`.
Entrypoint portable là `scripts/training/train_qlora_vast.py`. Quy trình triển
khai gồm bước chuẩn bị checkpoint riêng, rồi preflight và chạy trainer:

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

Pilot trên 3090 đã được đo bên dưới. Chưa có bằng chứng đủ ngân sách 10 giờ/$5
để hoàn tất 1.305 batch còn lại; không ngoại suy một bước thành cam kết thời gian.


## Entrypoint và policy trên Vast

- Tạo venv Python 3.11 riêng, cài `cloud/vast/requirements.txt`; không sửa venv
  mặc định của template. Cần đúng phiên bản packages trong identity checkpoint.
- `--parent-checkpoint` luôn trỏ về checkpoint 695 đã xác minh và giữ nguyên
  `--parent-manifest-sha256`. Khi tiếp tục từ checkpoint Vast, đổi `--resume-from`
  sang checkpoint mới; các tham số cohort/parent giữ nguyên.
- Preflight dựng lại identity của parent từ trainer gốc, source plan và mã
  filtered continuation; so sánh toàn bộ config, math/runtime sources, packages,
  cursor và prefix. Loader vẫn kiểm mọi payload hash, tensor signature và RNG.
- Identity mới ghi hash cohort, mã migration, hooks, activation offload và backend.
  Publisher chỉ cho phép chuyển từ đúng manifest parent hoặc tiếp tục đúng child.
  Không nới kiểm tra identity để ép resume.
- Giữ NF4, r=32/alpha=64/dropout=0.05, FP16 compute, FP32 adapters/optimizer,
  loss và LR cũ. Một trajectory/GPU, global batch size 2, một update/batch.
- Bản triển khai đầu tiên giữ **CPU pinned activation offload** và nonreentrant
  gradient checkpointing. Dùng hai replica, trung bình gradient thủ công trước
  unscale/clip/update, không cấp phát DDP reducer/buckets. RAM precheck chọn chạy
  song song hoặc tuần tự; đây là ước lượng, không bảo đảm tuyệt đối không OOM.
- `--stop-at-step 696` chạy đúng một batch sau 695 và lưu trước khi kết thúc.
  Bỏ flag này để chạy đến 2.000 hoặc đến giới hạn thời gian, tùy mốc đến trước.
  Flag pilot không đổi identity hoặc lịch dữ liệu.
- Checkpoint mỗi 5 update, giữ 1 local và 3 trên HF khi truyền retention flags.
  Ghi hoàn chỉnh local trước khi publish; publish lỗi vẫn giữ checkpoint local
  và dừng. Xóa bản cũ sau publish thành công; retention HF tiếp tục squash history
  trong cùng repo/prefix chuyên dụng theo policy đã chấp thuận.

Ví dụ pilot, chạy từ root repo trên Vast sau khi có dữ liệu, checkpoint và cache
model gốc. Bỏ `--preflight-only` để chạy GPU. Không chạy hai writer cùng lúc:

```bash
/workspace/first-run-venv/bin/python scripts/training/train_qlora_vast.py \
  --index data/processed/first_run_epoch_sft/index.json \
  --source-plan cloud/kaggle/recovery/continuation-plan.json \
  --cohort data/processed/first_run_diverse_4000/diversity-plan.json \
  --parent-checkpoint hub-recovery/first-run/checkpoint-00000695 \
  --parent-manifest-sha256 2cdd8808a886c3f32ac887b3411ad8dae3d91ccb8363a6206c48bf8c735c2b95 \
  --output 'outputs/training/first run/vast-pilot' \
  --checkpoint-dir 'outputs/training/first run/vast-checkpoints' \
  --checkpoint-hub-repo hqcfmlno1/QwenStacy-4b-coding \
  --checkpoint-hub-repo-type dataset --checkpoint-hub-prefix first-run \
  --checkpoint-hub-keep-last 3 --keep-checkpoints 1 --save-every-steps 5 \
  --max-train-seconds 3600 --stop-at-step 696 --preflight-only
```

Sau pilot, dùng thư mục output **mới cho mỗi phiên**, `--resume-from` trỏ vào
checkpoint Vast mới nhất và bỏ `--stop-at-step`. Dữ liệu và checkpoint dùng chung;
log tách theo phiên để monitor không đọc nhầm lỗi của phiên trước. Launch dưới
Supervisor để SSH/browser mất kết nối không giết tiến trình. Thoát trainer không
đồng nghĩa stop instance; GPU vẫn tính tiền cho tới khi instance được stop.

Log trong output mỗi phiên: `first run.log` (torchrun), `rank0.jsonl` và
`rank1.jsonl` (events/loss/thời gian/VRAM), `preflight.json`, rank summaries và
`report.json` khi hoàn tất. `telemetry/progress.log` là bản ngắn mỗi phút,
`telemetry/status.json` là trạng thái gần nhất, `telemetry/memory-*.jsonl` giữ
chi tiết RAM. Monitor dừng descendants khi phát hiện rank failed hoặc host OOM.

**Mạng của host ngày 19/10 (lịch sử):** HF API dùng CONNECT relay qua máy local;
`HTTPS_PROXY=http://127.0.0.1:18082`, `HF_HOME=/workspace/.hf_home`. Giữ máy local
và tunnel hoạt động để publish HF. Chạy dưới Supervisor chỉ độc lập với phiên
SSH điều khiển; nó chưa độc lập với relay mạng này. Mất relay có thể làm publish
lỗi và dừng tại checkpoint local. Không coi đây là backup HF chạy tự chủ qua đêm.
Không gửi token vào chat, git, command line hay log. Token chỉ nhập qua HF CLI.

Để yêu cầu dừng an toàn, tạo file `STOP` trong output của đúng phiên đang chạy
(ví dụ `touch "outputs/training/first run/vast-pilot/STOP"`). Hai rank sẽ kiểm tra
ở ranh giới optimizer, lưu trạng thái hoàn chỉnh rồi dừng. Không tái sử dụng
output còn file `STOP` cho phiên mới.

## Kết quả thực đo trên 2×RTX3090 ngày 19/10/2026

- Pilot resume 695 → 696, một trajectory/GPU: 26.604 và 18.976 token,
  một update mất 59,2 giây; peak allocated 12,69/10,16 GiB. Lưu và publish
  checkpoint mất khoảng 105 giây, tính riêng với thời gian update.
- Phiên train chính sau pilot dùng global batch 2, checkpoint mỗi **20** update,
  giữ 1 bản trên Vast và 3 bản trên HF. Ngân sách phiên là 8 giờ; wrapper yêu cầu
  stop instance khi trainer thoát. Stop instance vẫn phát sinh phí lưu đĩa.
- Thử riêng global batch **4**, microbatch **2/GPU**, accumulation **1**, từ
  checkpoint **700**: chọn 4 trajectory dài nhất **chưa train** là 27.997,
  27.993, 27.993 và 27.991 token. Mỗi GPU chạy một forward có batch dimension 2;
  right padding bị loại khỏi loss. Loss là trung bình đều của 4 trajectory.
- Đúng **một** update hoàn tất, raw time **107,79 giây**, loss **0,89853**,
  gradient norm **0,92344**. Gradient và adapter hữu hạn; hash adapter sau update
  giống nhau giữa hai GPU. Peak allocated **22,423/22,420 GiB**, peak reserved
  **23,078/23,078 GiB**. Giữ CPU pinned offload và toàn bộ cấu hình số học cũ.
- Thử nghiệm không ghi checkpoint hoặc publish; bỏ trọng số thử và resume
  phiên chính từ 700 với global batch **2**. Một bước thành công chưa chứng minh
  đủ headroom cho cả lượt train, cũng chưa đo được mức tăng throughput trên cùng
  tập mẫu. Script và báo cáo riêng nằm trong output `vast-batch4-pilot/`.

## Phiên chính chuyển sang global batch 4

Theo yêu cầu sau pilot, phiên batch 2 đã lưu và dừng ở checkpoint **710**:
`global_step=710`, `next_batch=710`, `samples_consumed=1420`. Entrypoint mới
`scripts/training/train_qlora_vast_batch4.py` và module `vast_microbatch.py`
được sửa ở local rồi chép bằng SCP qua SSH lên server; tại thời điểm triển khai
các file này chưa commit/push. Hash SHA256 của hai bản đã được đối chiếu.

- **2 trajectory/GPU**, world size **2**, global batch **4**, accumulation **1**.
  Vẫn NF4, r=32/alpha=64, dropout=0,05, LR=5e-5, FP16 compute, FP32 adapter và
  optimizer, CPU pinned activation offload, không cấp phát DDP buckets.
- Chỉ đúng manifest checkpoint 710 được phép chuyển identity; parent phải khớp
  identity của launcher batch 2 dựng lại từ các nguồn/config/package cũ. Loader
  nguyên bản kiểm payload hash, tensor signatures, optimizer, scaler và RNG.
  Child identity ghi config batch mới, hash launcher/module và manifest parent;
  các lần resume child tiếp theo phải khớp toàn bộ identity này.
- Giữ nguyên thứ tự và 1.420 mẫu đã train, ghép **phần còn lại** theo nhóm 4,
  right pad và loại padding khỏi loss. Mỗi trajectory được chuẩn hóa riêng,
  sau đó lấy trung bình đều toàn global batch; tail nhỏ hơn 4 vẫn dùng đúng hệ số.
- Còn **2.580 trajectory / 645 update**, vì vậy kết thúc 4.000 trajectory tại
  **global step 1355**. Từ sau 710, không còn suy ra số trajectory bằng `2×step`;
  đọc trường `samples_consumed` trong log/manifest.
- Lưu ngay sau update đầu tiên theo config mới, rồi mỗi **10 update** (40 mẫu),
  giữ 1 child checkpoint local và 3 trên HF. Parent 710 được pin riêng tại
  `hub-recovery/batch4-parent/checkpoint-00000710` để kiểm tra các lần resume.
- Kế thừa thời hạn 8 giờ của phiên cũ, trừ thời gian đã chạy và thời gian chuyển;
  không khởi động lại ngân sách 8 giờ. Giữ wrapper stop instance khi trainer thoát.
- Log server: `outputs/training/first run/vast-batch4/`; bản mirror local có
  `progress.log` và `first run.log` cùng thư mục. HF vẫn cần relay từ máy local.
- Kiểm tra trước triển khai: **178 passed, 3 skipped**; gồm kiểm thứ tự/cursor,
  loss và gradient của padded batch so với trung bình trajectory, khôi phục
  checkpoint đầy đủ và từ chối identity khác. VRAM vẫn sát trần như pilot;
  một bước thử thành công không bảo đảm toàn phiên không OOM.
- Xác nhận sau triển khai: step **711** hoàn tất, `samples_consumed=1424`, raw
  **80,96 giây**, loss **0,83556**. Lưu/publish mất **99,49 giây**; HF `latest.json`
  trỏ đến `checkpoint-00000711` và manifest khớp checkpoint local đã kiểm payload.
  Hai rank tiếp tục forward step **712**. Đây là kiểm tra một bước đầu, chưa phải
  kết luận ổn định cho toàn bộ 645 update.

## Gia hạn phiên đến sáng 20/10

Theo yêu cầu lúc khoảng 01h35 ngày 20/10 (UTC+7), đã yêu cầu dừng ở optimizer
boundary để lưu ngay. Hai rank hoàn tất và publish checkpoint **863**, progress
`global_step=863`, `next_batch=863`, `samples_consumed=2032`; HF latest và manifest
đã được đối chiếu với checkpoint Vast có payload được kiểm hash.

Phiên tiếp theo resume đúng checkpoint 863, giữ global batch 4, cadence 10 step
và retention cũ. Đổi thời hạn từ giới hạn 8 giờ ban đầu sang **07h45 ngày 20/10,
giờ Bangkok/Việt Nam**. Launch budget là 21.923 giây, `stop_margin_seconds=120`,
`expected_step_seconds=180`; trainer ngừng nhận batch mới sớm vài phút để có thời
gian lưu/publish rồi wrapper yêu cầu stop instance. Thời gian lưu phụ thuộc mạng;
đây là dừng có kiểm soát, không phải lịch cưỡng chế tắt instance đúng từng giây.

Output mới: `outputs/training/first run/vast-batch4-until-0745-20261020/`.
Alias `outputs/training/first run/current/` trên Vast trỏ đến output này; xem log
gọn qua `current/telemetry/progress.log`. Alias local cùng tên có bản mirror
`current/progress.log` và `current/first run.log`; hai file log local ở thư mục
`vast-batch4/` cũ cũng tiếp tục nhận log phiên mới để giữ lệnh tail cũ hoạt động.
Receipt triển khai: `outputs/training/first run/deadline-0745/receipt.json` trên
Vast, bản local ở `vast-batch4-until-0745-20261020/deployment.json`.

Để có lệnh stop độc lập với trainer, đã arm chương trình Supervisor
`first-run-deadline-stop` trên Vast. Nó chờ tới **07h45 ngày 20/10 UTC+7** rồi gọi
Vast API để stop chính instance, qua đường mạng trực tiếp không dùng relay local.
Nếu API lỗi/timeout, nó thử lại và Supervisor khởi động lại nếu hết lượt mà thất bại.
Trainer vẫn có thời gian dừng/lưu sớm như trên; watchdog là mốc dự phòng khi
trainer chưa thoát. API được gọi tại mốc giờ đó, thời điểm instance thực sự chuyển
sang stopped phụ thuộc thời gian xử lý của Vast. Không cần người dùng dậy stop tay.

Script vận hành: `/workspace/first-run-deadline-stop.py`; cấu hình
`/etc/supervisor/conf.d/first-run-deadline-stop.conf`; log có cấu trúc:
`outputs/training/first run/deadline-0745/watchdog.jsonl`.
Chương trình được start thủ công cho phiên hiện tại, `autostart=false` để lần
thuê/khởi động instance tương lai không tự áp dụng lại deadline đã hết hạn.

Cập nhật lúc khoảng 02h00 ngày 20/10: khi khảo sát thay lịch 07h45 bằng hủy
instance theo số dư, đã dừng `first-run-deadline-stop`. Container API key thiếu
`user_read` và `billing_read`, nên chưa triển khai hoặc bật bộ giám sát số dư hay
cơ chế tự hủy. Người dùng sau đó chọn tự thức dậy thao tác trên Vast. Trainer
vẫn chạy với ngân sách 21.923 giây và margin cũ, do đó vẫn có cơ chế lưu/dừng
có kiểm soát gần 07h45; watchdog cưỡng chế độc lập hiện **STOPPED**. Wrapper
vẫn yêu cầu **stop** khi trainer thoát. Người dùng cần **Destroy** instance để
chấm dứt phí đĩa sau khi xác nhận checkpoint đã có trên HF; Stop vẫn tính phí đĩa.

## Tiếp tục trên instance 57065621, ngày 21/10

Server mới: 2×RTX3090, RAM cgroup khoảng 181 GiB. Khôi phục đầy đủ từ
checkpoint 1000 (2.580 trajectory), tiếp tục đúng optimizer/scaler/RNG đã lưu.
Launcher `resume_qlora_vast_batch4.py` kiểm identity của checkpoint batch 4
đã có; không cần tải lại payload checkpoint 710 đã bị retention xóa khỏi HF.
Manifest tổ tiên vẫn được kiểm checksum và chỉ dùng làm metadata đối chiếu.

Theo yêu cầu học tiếp pool chưa train, launcher `train_qlora_vast_pool.py` giữ
nguyên toàn bộ thứ tự 4.000 mẫu cũ, nối thêm 5.966 mẫu từ pool đã lọc. Ưu tiên
mẫu của task chưa xuất hiện rồi cân bằng repo; mỗi trajectory chỉ xuất hiện
một lần. Tổng pool: **9.966 trajectory, 5.108 task, 1.501 repo**, 210.967.717 token.
Phần chưa train vẫn **dưới 28.000 token**. Plan nằm tại
`data/processed/first_run_diverse_remaining/pool-plan.json` (không đưa data vào git).

Chuyển lịch tại checkpoint **1003**, đã học **2.592 trajectory**. Chỉ đúng
manifest SHA256 `027d8f2cda5f7ad54a7110eb0691b5d8ec720663b53d23a39cd11eb9a435fc41`
được phép đổi identity. Child lưu metadata parent để tiếp tục resume khi HF
retention xóa payload parent. Không đổi numerical worker, loss, LoRA hay optimizer.
Global batch vẫn 4, hai trajectory/GPU; không lặp lại dữ liệu đã học. Mốc 4.000
mẫu vẫn là step 1355; toàn pool kết thúc ở **step 2847**, batch cuối có hai mẫu.

Không đặt deadline theo giờ; tham số tương thích launcher là 315.360.000 giây.
User xác nhận auto billing đã tắt/không lưu thẻ. Vast tự stop khi hết số dư;
không có watcher đọc số dư vì container key thiếu quyền billing. Không bảo đảm
lưu được update đang chạy ngay lúc Vast cắt máy; checkpoint HF gần nhất là điểm
khôi phục. Save ngay update đầu sau chuyển lịch, rồi mỗi 10 update; giữ 1 local,
3 HF và squash history. Supervisor chạy độc lập SSH; wrapper yêu cầu stop
instance khi trainer hoàn tất hoặc thoát lỗi. Stop vẫn phát sinh phí đĩa.

Output trên server: `outputs/training/first run/vast-pool-20261021/`, alias
`outputs/training/first run/current/`. Xem log gọn trên server bằng:

```bash
tail -n 20 -F '/workspace/Workflow-Aware-Agent-Distillation/outputs/training/first run/current/telemetry/progress.log'
```

HF được upload trực tiếp từ server này. Mirror log về máy cá nhân chỉ để xem,
không tham gia training hay upload checkpoint.

Xác nhận sau chuyển lịch: step **1004** hoàn tất, `samples_consumed=2596`, loss
**0,9097845**, khoảng **203 giây/update**. Lưu/publish checkpoint mất **78,64 giây**.
HF `latest.json` trỏ `checkpoint-00001004` tại revision
`6dcd23ae18e78732cd64ff209be756ed4901aec3`; manifest SHA256
`700c1b953f78880f0bb3122f6909c55b5ed038b75358be142cb33994ee7f6a2d`
khớp bản server đã kiểm hash toàn bộ payload. Hai rank tiếp tục forward step
**1005**, Supervisor RUNNING. Bộ kiểm tra local: **192 passed, 3 skipped**;
hash bốn source launcher/worker/pool trên local và server đã đối chiếu khớp.
