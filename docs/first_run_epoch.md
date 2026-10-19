# First run: QLoRA một epoch trên hai T4

Tên trong log: **`first run`**. Trainer gốc:
`scripts/training/train_qlora_epoch.py`; continuation đã lọc dùng
`scripts/training/train_qlora_filtered.py`. Cấu hình số học nằm tại
[`qlora_current_config.md`](qlora_current_config.md) và
`src/workflow_distillation/qlora_runtime.py`.

Trạng thái xác nhận ngày 2026-10-18: cấu hình r=32/alpha=64, PagedAdamW32bit đã
resume trên hai T4 và publish checkpoint **375** (750 trajectory). Recovery 6
đã dọn cache nhưng vẫn host RAM OOM trong update **371**, với hai trajectory
29.671 và 26.737 token. Recovery **version 8** thêm lịch song song/lần lượt theo
RAM; lượt resume tiếp theo báo CUDA OOM trong MLP/LoRA dropout ở backward.
Recovery **version 9** bỏ DDP reducer/bucket ngay khi khởi tạo và đã hoàn tất
update 371–376 trên CUDA; checkpoint 375 upload thành công. Update 377 lỗi
CUDA OOM trong backward rank 0 với trajectory 29.991 token; rank 1 có 28.704
token, lịch serial, host `oom_kill=0`. Watchdog dừng workers, exit 86.
Recovery **version 10** giữ backend này và lọc **chỉ phần chưa train** từ mốc
375 xuống **dưới 28.000 token**. Lịch mới gồm 9.966 trajectory/4.983 update.
Chưa hoàn tất epoch, chưa đánh giá chất lượng hoặc chứng minh mọi sample dưới
28k đều vừa bộ nhớ. Các mục điều tra bên dưới ghi lại trạng thái từng thời điểm;
hướng dẫn hiện tại ở mục **Recovery 10** cuối tài liệu.

## Dữ liệu và giới hạn epoch

Chuẩn bị bằng CPU, không load model:

```bash
.venv/bin/python -u -m scripts.data.prepare_epoch_sft \
  > 'outputs/training/first run/data-preparation.log' 2>&1
```

Chọn **tất cả** record `partition=train_pool`, `resolved=1`, `token_count<30000`
từ `data/processed/nvidia_mini_swe_agent_v1_2_success_qwen35_4b`.
Không giới hạn vào tập 4.000 đã chọn, không lấy validation/excluded, không cắt
trajectory. File đầu ra `data/processed/first_run_epoch_sft/index.json` chỉ xuất
hiện khi chuẩn bị thành công; hai file `tokens.bin` (uint32) và `groups.bin`
(uint8) lưu cùng thư mục. Index chứa số record/token thực tế, SHA-256 dữ liệu,
tokenizer, nguồn, loss weights và denominator từng trajectory.

Lần chuẩn bị ngày 2026-10-17 đã hoàn tất: **11.527 trajectory**, **256.253.032
token**, độ dài **5.149–29.997**, thuộc 5.792 task; tổng **5.764 update** ở global
batch size 2. Ba file dữ liệu chiếm khoảng **1,20 GiB**. SHA-256 của index:
`d0103eb4eaa2aed12e0d85cc9b952b54ee6a0c9d6536c365fd2a67296a7ccd0e`.
Đây là số lượng sau khi giữ đúng `train_pool`; không đưa trajectory held-out
vào train dù trajectory đó thành công và đủ ngắn.

Script từ chối ghi đè thư mục chuẩn bị đã tồn tại. Không khởi chạy hai bản chuẩn
bị đồng thời. Nếu lần chuẩn bị thất bại, kiểm tra log và dùng thư mục đầu ra mới
sau khi đã giải quyết nguyên nhân; không dùng các binary chưa có index hoàn chỉnh.

Shuffle cố định seed 42, batch size 1/GPU, hai GPU, accumulation 1, global batch
size 2. Tổng số optimizer update là `ceil(records/2)`. Batch cuối lẻ giữ sample
thật với trọng số đúng; rank còn lại tham gia backward với hệ số loss bằng 0.
Resume tiếp tục từ batch kế tiếp đã lưu, không bắt đầu lại epoch.

## Lưu checkpoint với ổ đĩa local khoảng 30 GB

- Lưu ngay sau update đầu tiên, sau mỗi 5 update, cuối epoch và khi dừng có kiểm soát.
- Giữ 1 checkpoint local sau khi ghi và upload thành công. Ghi bản mới vào staging
  trước khi xóa bản cũ, nên tạm cần đủ chỗ cho **hai** bản đầy đủ.
- Một checkpoint DDP r=32/FP32 ước tính **1,2–1,3 GiB**: adapter một bản, optimizer,
  GradScaler, RNG và cursor của cả hai rank. Kích thước thực tế phải đo trên GPU.
- Không lưu lại base model trong checkpoint. Binary training data 5 byte/token
  cộng index; không tải base model về máy local khi chuẩn bị.
- Upload lỗi làm dừng train và giữ bản local mới cùng bản trước; không xóa bản
  trước rồi tiếp tục khi chưa backup được.

KRun có thể lưu thêm bản artifact theo từng job trong `.krun/jobs/`. Chính sách
của trainer không dọn các bản tải về từ job cũ; sau khi đã xác minh bản mới và
backup Hub, chỉ giữ một bản recovery local qua các phiên. Không tải lại toàn bộ
artifact nhiều lần hoặc để mỗi phiên giữ một bản GiB trên máy 30 GB trống.

Chính sách giữ 1 bản chỉ giới hạn **local**. Mặc định publisher lưu mọi checkpoint
đã upload trong repo HF và lịch sử. Epoch trainer có tùy chọn giữ số bản cloud
và xóa lịch sử, nhưng mặc định tắt và chỉ bật sau khi chủ repo chọn rõ ràng.
Xem [chi tiết checkpoint](training_checkpoint_resume.md).

## Tạo nơi backup trên Hugging Face

1. Đăng nhập, mở <https://huggingface.co/new>, chọn **Model**, đặt tên ví dụ
   `qlora-first-run`, chọn **Private**. Nên tạo repo riêng cho lượt train này.
2. Mở <https://huggingface.co/settings/tokens>, tạo token **fine-grained**, cấp
   quyền đọc và ghi vào đúng repo vừa tạo. Không đưa token vào Git hoặc chat.
3. Trong giao diện notebook Kaggle, mở **Add-ons → Secrets**, tạo secret tên
   `HF_TOKEN`, dán token vào Value và bật quyền sử dụng cho notebook. Với notebook
   do KRun tạo, KRun sẽ lấy secret bằng `--secret HF_TOKEN`; nếu Kaggle yêu cầu
   liên kết secret với notebook, cần bật secret cho đúng notebook đó.
4. Gửi cho người vận hành **tên repo** dạng `USER/qlora-first-run` và **tên secret**.
   Không gửi giá trị token. Cấu hình backup phải được chốt trước khi gửi job dài.

Epoch trainer và công cụ fetch cũng hỗ trợ **Dataset repo private**. Nếu đã tạo
repo Dataset thì không cần tạo lại: thêm `--checkpoint-hub-repo-type dataset`
vào trainer và `--repo-type dataset` khi fetch. Repo của lượt này là
`hqcfmlno1/QwenStacy-4b-coding` (Dataset), Secret `HF_TOKEN`.

Chính sách người dùng đã chọn cho **first run**: giữ **3 checkpoint mới nhất
trên Hugging Face**, **1 checkpoint mới nhất trên máy cá nhân**. Dùng
`--checkpoint-hub-keep-last 3 --keep-checkpoints 1`; việc giữ 3 bản bao gồm dọn
lịch sử cũ trong repo riêng như mô tả bên dưới. Lưu sau update đầu tiên, mỗi
5 update và khi dừng có kiểm soát. `--keep-checkpoints 1` chỉ giới hạn thư mục
checkpoint trên **ổ đĩa phiên Kaggle**; chưa có tự đồng bộ về máy cá nhân.
Máy cá nhân hiện đã kiểm tra được checkpoint 5, chưa có checkpoint 215.
Khi đang ghi/upload có thể tạm giữ hai bản local để bản trước vẫn dùng được
nếu lần lưu mới lỗi; bản tải ban đầu trong `hub-recovery/` nằm riêng.

Trước khi cấp GPU, `scripts/cloud/checkpoint_preflight.py` chạy trên **CPU** để
tìm đúng dataset mount, kiểm tra index/binary SHA-256 và quyền ghi bằng
`HfApi.auth_check(write=True)`. Không tạo Hub commit, không xóa lịch sử, không
load model. Kaggle cấp quyền Secret theo notebook; nếu Secret chưa được bật cho
notebook KRun, preflight chờ tối đa 20 phút và chỉ log tên Label/trạng thái.
Bật `HF_TOKEN` trên đúng notebook đó; không cần tạo lại token. Dùng lại chính
kernel ID đã kiểm tra cho phiên GPU để giữ liên kết Secret, không dùng một
kernel mới mà chưa cấp quyền. Giữ log/report CPU trước khi chuyển phiên GPU.

Theo [tài liệu storage của HF](https://huggingface.co/docs/hub/storage-limits)
kiểm tra ngày 2026-10-17: tài khoản free có **100 GB private storage**, PRO có
**1 TB**. Đây là quota tổng của tài khoản, phải trừ phần đã dùng. Với checkpoint
ước tính 1,2–1,3 GiB, tài khoản free trống có thể chứa khoảng **70 bản đầy đủ**
khi còn dự phòng; PRO khoảng **700 bản**. Đây là ước lượng dung lượng, không phải
giới hạn số checkpoint do Hub quy định. Đo file thật sau lần lưu đầu tiên để tính lại.

**Ghi đè hoặc xóa file khỏi phiên bản hiện tại chưa đủ để giải phóng lịch sử**.
Nếu upload mỗi 5 update trong cả epoch, tổng dung lượng lịch sử có thể vượt quota
free rất xa. Có hai phương án cần chọn trước khi chạy:

- Giữ 2 checkpoint mới nhất trong repo riêng và dọn cả lịch sử cũ bằng tùy chọn
  **`--checkpoint-hub-keep-last 2`**. Tùy chọn này gọi `super_squash_history`,
  **xóa lịch sử main không thể hoàn tác**, cần chủ repo cho phép rõ ràng. Hai bản
  được giữ còn đầy đủ trong tree hiện tại trước khi squash. Script từ chối repo
  có file của run khác, branch/tag/PR refs khác; chỉ dùng một writer. API squash
  không hỗ trợ compare-and-swap, nên không chạy writer khác trên repo này.
  HF cho biết quota có thể mất **tới 36 giờ** mới phản ánh việc dọn lịch sử;
  không hứa tổng dung lượng được tính phí luôn chỉ bằng 2 bản. Phải chừa ngân
  sách upload trong thời gian này hoặc giãn lưu cloud/dừng trước khi đầy quota.
- Giữ lịch sử: đặt ngân sách upload hữu hạn, kiểm tra dung lượng và dừng trước
  giới hạn. Không thể cam kết lưu mọi checkpoint của cả epoch trong 100 GB.

Mặc định không xóa lịch sử. Upload/xóa lịch sử lỗi làm dừng và giữ checkpoint local;
nếu upload thành công nhưng squash lỗi, retry đúng checkpoint sẽ hoàn tất dọn
lịch sử mà không upload lại file lớn. Không áp dụng cho repo đã có dữ liệu khác.

## Khởi chạy và theo dõi

Chỉ upload ba file dataset đã chuẩn bị vào một **private Kaggle dataset**. Không
gắn binary lớn bằng `--input` vì package KRun có giới hạn dung lượng. Tên dataset,
đường dẫn `/kaggle/input/.../index.json`, repo HF và chính sách cloud phải được
điền vào manifest trước khi dùng mẫu sau:

```bash
krun run scripts/training/train_qlora_epoch.py \
  --gpu T4 --requirements cloud/kaggle/weighted-requirements.txt \
  --dataset USER/PRIVATE_EPOCH_DATASET --secret HF_TOKEN \
  --output 'outputs/training/first run' --cell-timeout 43200 --detach -- \
  --index /kaggle/input/PRIVATE_EPOCH_DATASET/index.json \
  --run-name 'first run' --max-train-seconds 39600 \
  --save-every-steps 5 --keep-checkpoints 1 \
  --checkpoint-hub-repo USER/qlora-first-run \
  --checkpoint-hub-prefix first-run \
  --stop-margin-seconds 600 --expected-step-seconds 240
```

Đây là mẫu tham số; cloud retention phải được giải quyết trước khi thực thi.
Nếu chủ repo đã chọn rõ phương án xóa lịch sử trong repo riêng, thêm
`--checkpoint-hub-keep-last 2` sau dấu `--`. Không thêm cờ này khi chỉ được phép
lưu backup nhưng chưa được phép xóa lịch sử.
Phiên GPU đầu tiên đã được gửi vào version 5 của **cùng notebook**
`cuonghoangquoc/workflow-aware-agent-dis-20261016-185315-746e15`, job KRun
`20261016-192211-b2bc75`, theo yêu cầu bắt đầu GPU ngay khi CPU/log API còn kẹt.
Đây là một lần gửi train thật, không phải pilot. Tham số thực tế nằm trong
`outputs/training/first run/gpu-job.json`; repo Dataset dùng
`--checkpoint-hub-repo-type dataset`, retention cloud 3/local 1. Mẫu `krun run`
ở trên tạo notebook mới; lần gửi thực tế dùng lại kernel ID hiện tại để giữ
liên kết Secret.

Kết quả thực tế của version 5: cài dependencies xong nhưng KRun dừng khi gọi
`UserSecretsClient.get_secret("HF_TOKEN")` với HTTP 400/ConnectionError. Trainer
chưa được gọi, model chưa load, **0 optimizer update**, chưa có checkpoint.
Log đầy đủ lưu tại `outputs/training/first run/gpu-remote-live.log` và được
chép nối vào `first run.log`. Kiểm tra lại dịch vụ/Label Secret trước khi dùng
thêm quota GPU; không suy luận lỗi token HF hoặc VRAM từ lỗi này. Phiên CPU
chẩn đoán version 6 cũng nhận HTTP 400 với response body không cung cấp chi tiết.

Một phiên dùng ngân sách worker 11 giờ, trừ dự phòng 10 phút và thời gian dự kiến
của step tiếp theo. Ngân sách này bắt đầu trước model load trong worker, không
gồm cài dependency/download ở launcher. Không phải thời gian cam kết hoàn thành
epoch và không đọc quota Kaggle còn lại. KRun cell timeout phải chừa thêm thời
gian launcher chuẩn bị. Một epoch dài có thể cần nhiều phiên nối tiếp từ checkpoint.
Không chạy thêm pilot hoặc tự retry cấu hình khác khi OOM.

Xem stdout/stderr trong tab **Logs** của notebook Kaggle, hoặc dùng endpoint
stream của Kaggle CLI:

```bash
kaggle kernels logs \
  cuonghoangquoc/workflow-aware-agent-dis-20261016-185315-746e15 --follow
```

`krun logs JOB_ID --follow` còn theo dõi trạng thái và tải artifact khi kết thúc.
Bản KRun hiện tại đọc log snapshot theo chu kỳ; snapshot có thể trống khi phiên
còn chạy, trong khi stream hoặc tab Logs đã có dữ liệu. Lỗi đọc log 500 từ
Kaggle không có nghĩa training đã dừng; kiểm tra trạng thái và kết nối lại phần
đọc log, không gửi lại job train. Output gồm:

| File | Nội dung |
| --- | --- |
| `first run.log` | stdout/stderr của hai training rank, ghi nối tiếp khi có dòng mới |
| `rank0.jsonl`, `rank1.jsonl` | Config, model load, forward/backward/optimizer, loss, thời gian step, VRAM, checkpoint và lỗi |
| `launcher.jsonl` | Kiểm tra dataset và kết quả launcher |
| `rank*-summary.json`, `report.json` | Trạng thái kết thúc, cursor và lỗi nếu có |
| `checkpoints/` | Checkpoint hoàn chỉnh mới nhất và `latest.json` |

Nếu không ghi được file log, launcher dừng cả nhóm worker và báo lỗi thay vì
tiếp tục train mà mất log.

File trên `/kaggle/working` được cập nhật trong phiên; muốn xem trên máy local
khi đang train dùng log remote. `--output` tải artifact khi phiên kết thúc, không
tự đồng bộ mọi file local theo từng dòng. Sau khi có bản local có thể dùng:

```bash
tail -f 'outputs/training/first run/first run.log'
```

`seconds` của mỗi step đo từ trước forward tới khi update, kiểm tra kết quả và
đồng bộ GPU xong; không cộng thời gian kết nối, cài đặt, model load hoặc checkpoint.

## Resume và kiểm chứng

### Khởi chạy từ giao diện sau lỗi Secret của KRun

Ảnh giao diện đã xác nhận `HF_TOKEN` được bật trên đúng notebook; lỗi HTTP 400
không đủ để kết luận người dùng nhập sai Label hoặc token. Version **7** đã
được lưu bằng `QUICK_SAVE` trên cùng kernel **139085360**, không gửi lượt chạy
GPU mới. API đọc lại đã xác nhận source hash
`e242682d348558e5cfbe4e0d41d399f8301c0c10a4ef0c1fe2e84ee5654c3bb5`,
entry point `scripts/training/train_qlora_epoch.py`, GPU `NvidiaTeslaT4`, Internet
bật và notebook private. Payload train và tham số giống version 5; chỉ chuyển
đọc Secret lên trước cài dependencies để lỗi được phát hiện sớm.

Mở lại [đúng notebook](https://www.kaggle.com/code/cuonghoangquoc/workflow-aware-agent-dis-20261016-185315-746e15/edit),
tải lại trang và xác nhận `ENTRYPOINT = 'scripts/training/train_qlora_epoch.py'`
thay cho `checkpoint_preflight.py`, rồi chọn **Save Version → Save & Run All**
với **GPU T4 x2**. Đây là thử khởi chạy qua giao diện để kiểm tra lỗi liên kết
Secret của đường API, chưa phải cách sửa đã được chứng minh. Sau khi người dùng
thao tác Secret và khởi chạy qua giao diện, **version 8** đã log `secret_loaded`,
kiểm tra dữ liệu thành công, load cả hai model replica và bắt đầu forward/backward
step 1 với 8.319/29.291 token. Số trainable đo trên mỗi rank là **64.929.792**.
Log stream được ghi vào `outputs/training/first run/browser-run-live.log`.
Chưa kết luận thao tác Save Version là nguyên nhân sửa lỗi Secret.
Step 1 đã hoàn tất trên cả hai rank: loss **0,66485679**, raw step **173,87/174,05
giây**. Checkpoint `checkpoint-00000001` đã log `published: true` và
`checkpoint_complete` trên cả hai rank, mất **39,57 giây** gồm ghi/upload;
phiên tiếp tục sang step 2. Bằng chứng lưu tại
`outputs/training/first run/browser-run-first-checkpoint.json`. Đây là xác nhận
save/upload thực tế, chưa kiểm chứng restore CUDA từ checkpoint đó.
Sau đó step 2 thất bại lúc `2026-10-16T19:51:38Z`: cả hai worker nhận
**SIGKILL / exit code -9**, launcher ghi `status: failed` lúc `19:51:40Z`.
Log không có bằng chứng xác nhận CUDA OOM hoặc host OOM; chưa kết luận nguyên
nhân hệ thống kill process. Kaggle API vẫn có thể báo `RUNNING` sau khi trainer
đã thoát, vì vậy không dùng trạng thái phiên làm bằng chứng train còn tiến triển.
Trạng thái recovery hiện tại là checkpoint step **1** đã publish; step 2 chưa
hoàn tất. Người dùng đã yêu cầu tiếp tục điều tra đến khi chạy ổn định; chưa
gửi lượt recovery GPU qua API vì cần giữ liên kết Secret của notebook hiện tại.
Không chạy đồng thời một phiên interactive và một phiên
Save & Run All vì cả hai cùng ghi vào một repo checkpoint.

Dùng `scripts/training/fetch_training_checkpoint.py` trên môi trường có quyền
đọc repo để tải checkpoint mới nhất; xem lệnh trong
[hướng dẫn resume](training_checkpoint_resume.md).
Checkpoint đầy đủ lớn hơn giới hạn package KRun, nên với epoch trainer cần đưa
nó vào private Kaggle dataset để gắn vào phiên mới, hoặc tải trực tiếp từ Hub
trong Kaggle trước khi gọi trainer. Không gắn checkpoint GiB bằng `--input`.
Chạy cùng entry point và index, thêm `--resume-from` trỏ tới thư mục checkpoint
hoàn chỉnh. Prefix epoch là **`first-run`**, không thêm `/ddp_once`.

Giữ nguyên model revision, code có hash, package/config, hai rank và dữ liệu
theo byte. Thay đổi khoảng lưu, nơi lưu hoặc ngân sách phiên không đổi objective.
Các test CPU kiểm tra resume state/RNG và DDP hai process với batch lẻ; chưa
kiểm chứng PagedAdamW32bit CUDA restore hoặc cloud upload thật. Chỉ báo hoàn tất
epoch khi `report.json` ghi `completed` và cursor bằng tổng số batch; trạng thái
`stopped_checkpointed` chỉ hoàn tất một phần epoch.

### Recovery sau SIGKILL ở step 2

Giả thuyết đang kiểm tra là bộ nhớ host pinned được cache sau backward của
step 1, rồi tăng thêm khi step 2 có độ dài khác. Đây **chưa phải nguyên nhân
đã xác nhận**. Lệnh `kaggle quota` sau lỗi báo còn **28,36/30 giờ GPU**;
không có bằng chứng quota tuần đã cạn tại thời điểm kiểm tra.

Gói recovery private đã sửa lỗi launcher là version **4** (xem chẩn đoán
và trạng thái xuất bản cuối tài liệu):
`cuonghoangquoc/first-run-memory-recovery-20261017/versions/4`.
Version 1 không dùng được vì Kaggle tự giải nén archive lúc nhập dataset;
version 2 giữ nguyên bytes archive dưới tên `original-source.bin`.
Không chứa token hoặc checkpoint. Chạy trong
[notebook hiện tại](https://www.kaggle.com/code/cuonghoangquoc/workflow-aware-agent-dis-20261016-185315-746e15/edit),
GPU T4 x2, với Secret `HF_TOKEN` đã liên kết:

```python
import kagglehub, runpy
from pathlib import Path

bundle = Path(kagglehub.dataset_download(
    "cuonghoangquoc/first-run-memory-recovery-20261017/versions/4"
))
runpy.run_path(str(bundle / "resume_first_run.py"), run_name="__main__")
```

Chọn **Save Version → Save & Run All** một lần. Không khởi chạy đồng thời
phiên interactive khác. Bootstrap từ chối nếu thư mục recovery đã tồn tại;
không xóa thư mục đó khi chưa kiểm tra có worker còn chạy hay không.

**Dataset recovery version 4 phải được gắn vào draft trước khi chạy nền.**
Nếu draft còn gắn version 2 hoặc 3, chỉ đổi chuỗi trong mã sang `versions/4` là
chưa đủ. Cập nhật dataset trong Input, hoặc chạy riêng cell sau trong phiên
interactive để KaggleHub gắn version 4 (không gọi bootstrap/train):

```python
import kagglehub
print(kagglehub.dataset_download(
    "cuonghoangquoc/first-run-memory-recovery-20261017/versions/4"
))
```

Chờ gắn xong, dừng Draft Session đang rảnh rồi mới Save & Run All.

Theo yêu cầu thay mã trực tiếp, notebook ID **139085360** đã được lưu ở
version **10** bằng `QUICK_SAVE`, gồm một code cell resume như trên. Đã đọc
lại mã và kiểm tra GPU, index dataset, privacy, Internet và Docker image.
Version 9 lưu Python dạng script khiến bước dựng preview của `QUICK_SAVE`
báo `NotJSONError`; version 10 lưu đúng định dạng notebook và dựng preview
thành công. Trạng thái `COMPLETE` của lượt lưu này chỉ xác nhận preview,
**không phải train hoàn tất hoặc recovery GPU đã chạy**. Public API không
cho kiểm chứng liên kết Secret. Cần tải lại editor để thấy bản mới, rồi
khởi chạy qua **Save Version → Save & Run All**.

`scripts/cloud/resume_first_run.py` kiểm tra SHA256 của archive, trainer và
index, tải checkpoint hoàn chỉnh mới nhất từ prefix `first-run` trên HF rồi
gọi trainer nguyên bản với `--resume-from`. Giữ nguyên source có hash, package,
dữ liệu, r32, optimizer, precision, objective và RNG; kiểm tra identity khi
restore vẫn bắt buộc. Việc restore PagedAdamW32bit trên CUDA chưa được kiểm
chứng bởi các test CPU.

`cloud/kaggle/recovery/sitecustomize.py` chỉ bật khi biến
`FIRST_RUN_HOST_CACHE_RECOVERY=1`. Hook đồng bộ CUDA, thu gom đối tượng Python
đã hết tham chiếu và gọi `torch._C._host_emptyCache()` trước mỗi context
`save_on_cpu`. Chỉ cache pinned đã rảnh được giải phóng; pinned activation
offload vẫn bật. Nếu API cần thiết không có, worker thoát với code 78 trước
khi train. API này là nội bộ của PyTorch và đã đối chiếu với source 2.11.0;
hiệu quả thực tế vẫn cần kiểm chứng bằng lượt GPU recovery.

Telemetry ghi mỗi 10 giây vào stdout và
`/kaggle/working/first-run-recovery/memory-<pid>.jsonl`: RAM cgroup và giới hạn,
bộ đếm `oom`/`oom_kill` nếu kernel cung cấp, RSS tiến trình, pinned allocated/
active/peak và CUDA allocated/reserved. Sự kiện `host_cache_cleared` ghi trước/
sau cleanup. `recovery-manifest.json` lưu hash hook và cursor checkpoint.
Log remote local được thu vào
`outputs/training/first run/recovery-gpu-live.log` khi phiên recovery chạy.

Test CPU đã xác nhận cleanup không đổi loss, gradient hoặc RNG trên bài test
dropout; kiểm tra đọc cgroup và fail-closed khi thiếu API. Full suite:
`python -m pytest -q` **113 passed, 3 skipped**. Đây không phải bằng chứng
recovery GPU đã ổn định. Chỉ đánh giá recovery sau khi vượt step 2, hoàn thành
nhiều batch có độ dài khác nhau, publish checkpoint mới và kiểm tra xu hướng
bộ nhớ. Không kết luận hoàn tất epoch từ một checkpoint thành công.

### Lượt resume interactive version 2 và chẩn đoán version 3

Người dùng đã bật lại Secret trên giao diện và chạy recovery version 2 bằng
`Run All` trong Draft Session. Lượt này tải checkpoint step 1 thành công và
đã cài hook trên hai rank, nhưng launcher thoát code 1 khi model đang load
khoảng `223/723` trọng số. Log người dùng cung cấp chỉ có
`CalledProcessError` của launcher; chưa có lỗi gốc từ worker. Snapshot cuối
trước lỗi có pinned memory bằng 0, `oom=0`, `oom_kill=0`; không chứng minh
hay loại trừ OOM phát sinh sau snapshot. Chưa đến restore state hoặc forward.

Sau khi Draft Session dừng và khởi động lại, cell chẩn đoán không tìm thấy
`first run.log` hay rank summaries. Bộ đếm cgroup bằng 0 và RAM khoảng 0,8 GB
trong phiên mới không phải bằng chứng về lần lỗi cũ. API log chỉ thấy version
10 đã lưu; không đọc được stdout của Draft Session. Không suy luận trainer
vẫn chạy chỉ từ trạng thái của saved version.

Recovery dataset version 3 giữ nguyên source archive và tham số train, thêm:

- Hook vận hành thêm `--log-dir .../torchrun --tee 3` cho đúng lệnh torchrun;
  giữ stdout/stderr riêng từng rank và console. `PYTHONFAULTHANDLER=1` giúp
  ghi lỗi native có thể xử lý; không bắt được SIGKILL.
- Telemetry chỉ rõ GPU của rank khi đo từ background thread, đồng thời ghi
  `memory.stat` (anon/file/shmem), peak RAM và giới hạn/bộ đếm số tiến trình.
- Khi launcher lỗi và bootstrap còn sống, thu tail log và bộ đếm **sau lỗi**
  vào `first-run-recovery/failure-diagnostics.json`, in ra console và upload
  phần chẩn đoán vào README của **repo dataset private hiện tại**. Giữ nội
  dung README trước đó, thay đúng mục chẩn đoán cũ; không sửa checkpoint
  hoặc latest pointer. README là đường dẫn được policy dedicated repo cho
  phép. Upload dùng parent revision và chỉ diễn ra sau launcher thoát để
  tránh xung đột với publication. Token được lọc khỏi bản chẩn đoán.

Backup này cần bootstrap còn chạy và mạng/token còn hoạt động; không bảo đảm
sống qua việc kill toàn container. Chạy **Save Version → Save & Run All**
qua giao diện sẽ tạo phiên background để agent theo dõi log qua API. Khi
Draft Session đang rảnh, dừng nó trước khi chạy background để không giữ GPU
cho hai phiên. Không ghi version notebook qua API vì đã có bằng chứng UI
cho thấy liên kết Secret bị mất sau các lần ghi trước.

Kiểm chứng local: hai worker CPU với lỗi cố ý ở rank 1 đã giữ được nguyên
thông báo lỗi trong `stderr.log`; kiểm tra redaction, giữ nội dung README,
private repo và parent revision. Full suite **119 passed, 3 skipped**;
Ruff và `git diff --check` pass. Lượt thực tế version 3 sau đó phát hiện lỗi
tương thích `Popen(args=...)`, được ghi rõ bên dưới.

Notebook **version 11** đã được người dùng Save & Run All qua giao diện.
API log xác nhận mã yêu cầu recovery dataset version 3 nhưng notebook vẫn
gắn version 2. KaggleHub báo `New Datasets cannot be attached in
non-interactive sessions. Version [3] requested, found version [2] attached`.
Lỗi ở cell 1, trước khi vào `resume_first_run.py`; chưa đọc Secret, download
checkpoint, load model hoặc train. Do bootstrap chưa chạy, cơ chế diagnostic
HF của version 3 chưa được gọi trong lượt này. Đây là lỗi chuẩn bị nguồn dữ
liệu, chưa phải bằng chứng về lỗi model load hoặc kết quả recovery GPU.

### Lỗi launcher version 3 và bản sửa version 4

Log đầy đủ do người dùng cung cấp lúc `2026-10-17T03:14:28Z` xác nhận:

```text
TypeError: install_torchrun_logs.<locals>.popen() missing 1 required positional argument: 'command'
```

Đây là lỗi trong hook ghi log bổ sung ở version 3. PyTorch 2.11 gọi
`Popen(args=...)` khi tạo worker; wrapper dùng tên tham số đầu là `command`,
nên Python từ chối lời gọi trước khi tạo worker. Checkpoint đã tải nhưng chưa
load model, restore optimizer hoặc train step mới. `CalledProcessError` và
`SystemExit: 1` là các lỗi bao ngoài. Cảnh báo YAML README không gây dừng train.

Bộ đếm **sau lỗi** có `oom=0`, `oom_kill=0`, memory `max=0` và pids `max=0`.
Peak cgroup khoảng 24,58 GiB trên giới hạn 30 GiB. Sau lỗi, khoảng 22,69 GiB
là file cache, anonymous memory khoảng 0,47 GiB. Không quy lỗi lần này cho
OOM. Lỗi SIGKILL step 2 ở lần train gốc và lỗi load model của recovery version
2 vẫn chưa xác định được nguyên nhân.

Version 4 đổi đúng chữ ký wrapper thành `popen(args, *popen_args, **kwargs)`,
giữ nguyên chuyển tiếp tham số, environment, cwd, log riêng hai rank và
command đầu vào. Source archive SHA256, tham số train, cấu hình QLoRA, index
và kiểm tra identity khi resume giữ nguyên. Test subprocess thật gọi bằng
keyword đã tái hiện đúng TypeError trước sửa và pass sau sửa. Bài test hai
worker CPU vẫn lưu được lỗi cố ý của rank 1. Toàn bộ test: **120 passed,
3 skipped**; Ruff và `git diff --check` pass. Chưa có bằng chứng recovery
version 4 chạy ổn định trên GPU.

Gói version 4 đã xuất bản; API xác nhận `ready` và `current_version_number=4`,
danh sách file remote khớp tên/kích thước của gói local. Bằng chứng nằm trong
`outputs/training/first run/recovery-v4-remote-status.json`,
`recovery-v4-remote-files.json` và `recovery-v4-assets.json`. Không dùng version 3
để resume lại. Gắn version 4 bằng cell download-only
ở trên trong draft, sửa cell resume sang version 4, dừng Draft Session đang
rảnh rồi Save Version → Save & Run All với T4 x2. Không xóa thư mục của phiên
cũ để chạy lại trong cùng session. Không ghi notebook qua API.

### Recovery version 4: xác thực HF, train tới step 5 và xung đột publication

Lượt đầu của version 4 dừng trước model load vì HF trả HTTP 401 ở
`repo_info`. Kiểm tra riêng `HfApi(token=Secret).whoami()` cũng trả 401,
xác nhận token đang lưu không được HF chấp nhận. Sau khi người dùng cập nhật
Secret, ảnh output xác nhận account `hqcfmlno1`, `read_repo: OK` và
`write_repo: OK`.

Log API của phiên background kế tiếp (worker PID 198/199) xác nhận hai rank load model, restore checkpoint 1 và
hoàn thành các update sau (số giây raw của rank 0):

| Step | Trajectory đã xử lý | Giây | Loss | Grad norm |
| --- | --- | --- | --- | --- |
| 2 | 4 | 166,17 | 0,722917 | 0,575839 |
| 3 | 6 | 122,89 | 0,594443 | 0,390571 |
| 4 | 8 | 128,63 | 0,601573 | 0,369814 |
| 5 | 10 | 120,84 | 0,730039 | 0,429369 |

Đây là 10/11.527 trajectory tính cả hai sample của checkpoint 1. Loop train
hiện ghi JSON `event=step`, `samples_consumed`, `global_step` và
`epoch_fraction`; chưa có tqdm cho epoch. Thanh lúc tải/load weights không
biểu thị tiến độ train. `epoch_fraction` trong log tính theo số update;
tiến độ trajectory có thể lấy `samples_consumed / 11527`.

Lúc `2026-10-17T03:57:38Z`, cả hai worker dừng khi publish checkpoint 5:
`Remote checkpoint is already at this step or newer`. Publisher kiểm tra
identity/world size trước, cho phép retry nếu hai manifest giống hệt nhau,
và từ chối manifest khác khi remote đã ở cùng step hoặc mới hơn. Không bỏ
guard hoặc ghi đè checkpoint remote để tiếp tục. Người dùng xác nhận
`first-run/latest.json` trên HF trỏ tới `checkpoint-00000005`; chưa có
manifest remote để so hash và chưa xác định writer nào đã tạo bản đó.

Checkpoint local 5 được commit trước publication và đã giữ trong output
Kaggle. Đã tải đầy đủ về
`outputs/training/first run/local-checkpoints/checkpoint-00000005`, kiểm tra
đúng identity của checkpoint 1, world size 2, progress step 5/batch 5/10 sample
và SHA256 của adapter cùng hai file state rank. Tổng 1.299.375.319 byte,
khoảng 1,21 GiB. Không xác nhận bản trên HF giống hệt bản local khi chưa so
manifest. Bằng chứng kiểm tra:
`outputs/training/first run/recovery-v4-local-checkpoint-verification.json`.

Bộ đếm sau lỗi có `oom=0`, `oom_kill=0`, memory `max=0`; peak cgroup khoảng
27,90/30 GiB. Version 4 đã vượt batch step 2 từng SIGKILL và chạy thêm nhiều
batch, nhưng chưa có lần publication checkpoint mới thành công được xác
nhận cho phiên này. Vì vậy chưa kết luận train và backup chạy ổn định.

Log realtime của phiên background đã lưu ở `outputs/training/first run/current-user-run-live.log`.
Luồng đọc từng gián đoạn vì DNS local không phân giải `api.kaggle.com`;
khôi phục bằng DNS fallback chỉ trong tiến trình đọc, cùng hostname/TLS và
thông tin đăng nhập. Việc mất luồng log không phải bằng chứng train lỗi;
lỗi publication được xác nhận từ event `failed` và traceback sau khi nối lại.

### Phân biệt phiên background đã lỗi với Draft đang chạy

Ảnh người dùng cung cấp sau đó cho thấy **Draft Session riêng vẫn đang chạy**
trên notebook `cuonghoangquoc/workflow-aware-agent-dis-20261016-185315-746e15/edit`:
GPU T4 ×2, có nút Cancel Run, worker PID 252/253 và log tới
`2026-10-17T04:09:25Z`. Console hiện upload
`checkpoint-00000010/rank-00000.pt` đạt 100%. Đây là bằng chứng tiến độ của
Draft, chưa xác nhận toàn bộ checkpoint 10 đã publish thành công.

Kết luận trước đó rằng phiên người dùng đang xem đã dừng là sai phạm vi:
API logs/status phản ánh phiên saved/background đã lỗi, không phản ánh
Draft interactive này. Các số step 5/10 trajectory ở trên thuộc phiên
background. Hai phiên cùng ghi vào một HF prefix là lời giải thích có cơ sở
cho xung đột publication, nhưng chưa đối chiếu manifest để xác nhận writer.

Giữ Draft hiện tại chạy; không Run All, Save & Run All, restart hoặc tạo
thêm lượt train. Không bỏ guard chống ghi đè checkpoint. Để xác nhận trọn
checkpoint 10, cần output của chính Draft như `checkpoint_saved` với
`global_step: 10` và `published: true`, hoặc bước train tiếp sau lần lưu
đồng bộ đó; thanh upload một file riêng lẻ chưa đủ.

### Version 15: OOM RAM hệ thống khi ghi checkpoint 205

Sau khi người dùng hủy phiên, output Kaggle cung cấp log từng rank và memory
telemetry. Cả hai rank đã hoàn thành update 205, `samples_consumed=410`, loss
1,603506 và bắt đầu `checkpoint_started` lúc `2026-10-17T12:58:16Z`. Không có
`checkpoint_complete` 205. Chỉ có file rank đang ghi trong thư mục staging;
không dùng thư mục đó để resume.

Checkpoint 200 đã hoàn tất lúc `12:46:08Z`, gồm publication, và con trỏ local
vẫn trỏ tới 200. Manifest có `complete=true`, world size 2, progress step/batch
200, 400 sample. Lượt tiếp theo tải checkpoint hợp lệ mới nhất trên HF và
tiếp tục update 201 nếu latest vẫn là 200, không bắt đầu epoch từ đầu.

Telemetry lúc `13:00:17Z` xác nhận `oom_kill=1`, `oom=26`, peak cgroup
32.212.258.816 byte so với giới hạn 32.212.254.720 byte (30 GiB). Trước khi
lưu, RAM khả dụng chỉ khoảng 0,76 GiB; mỗi rank giữ khoảng 9,66 GB allocator
RAM pinned. Đây là bằng chứng OOM RAM hệ thống trong giai đoạn ghi checkpoint,
trước upload HF. Chưa xác định PID bị kill. Cache pinned giữ sau backward là
nguyên nhân có cơ sở của thiếu headroom khi sao chép state sang CPU; chưa đo
trên GPU lượng cache thực tế được giải phóng bằng bản sửa mới.

Giả thuyết treo khi upload hoặc squash history HF được rút lại cho sự cố này.
Console UI và API đều dừng ở log checkpoint 130 trong khi log file tiếp tục
đến 205. Chưa chứng minh nguyên nhân cắt log công khai; không dùng đuôi log
đó để suy ra tiến độ hiện tại.

Recovery version 5 giữ nguyên archive, trainer, dữ liệu, cấu hình QLoRA và
identity checkpoint. Hook vận hành bổ sung CUDA synchronize, GC và
`torch._C._host_emptyCache()` trên **mọi rank trước `CheckpointManager.save`**,
bên cạnh cleanup trước forward đã có. Hàm chỉ giải phóng cache pinned không
còn hoạt động; không xóa optimizer, tensor đang dùng hay RNG. Full telemetry
10 giây/lần tiếp tục ghi vào `first-run-recovery/memory-<pid>.jsonl`, console
chỉ ghi event gọn; tắt progress bar HF để giảm lượng log lặp.

Kiểm thử CPU xác nhận cleanup xảy ra trước serialization và save/load tiếp
tục cho cùng loss, adapter weight, RNG/optimizer; kiểm thử subprocess xác nhận
cả hai hook được cài khi worker khởi động. Các kiểm thử này chưa chứng minh
GPU chạy ổn định dài hạn. Không mở lượt GPU mới trong lúc sửa.

Trong cùng notebook, gắn recovery dataset **version 5** bằng giao diện Kaggle
trước, giữ dataset epoch và Secret `HF_TOKEN` bật, rồi dùng cell:

```python
import kagglehub, runpy
from pathlib import Path

bundle = Path(kagglehub.dataset_download(
    "cuonghoangquoc/first-run-memory-recovery-20261017/versions/5"))
runpy.run_path(str(bundle / "resume_first_run.py"), run_name="__main__")
```

Chạy nền bằng **Save Version → Save & Run All** đúng một lần. Không chạy thêm
Draft song song. Log startup phải có `recovery_checkpoint_verified` ở mốc
mới nhất; khi tới checkpoint 205 cần thấy `host_cache_cleared` với
`phase=before_checkpoint`, rồi `checkpoint_complete` và update tiếp theo.

### Recovery 6: thoát khi host OOM kill và log tiến độ mỗi phút

Version 5 dọn cache trước checkpoint nhưng chưa có watchdog thoát sớm khi
cgroup có OOM kill mới. OOM killer gửi SIGKILL cho tiến trình được chọn;
không tạo exception Python để toàn bộ notebook tự kết thúc. Worker khác có
thể còn chờ collective, với timeout NCCL 1.800 giây. Timeout launcher là
`39600 + 900 = 40500` giây. Chưa xác định tiến trình bị kill trong version 15;
không khẳng định worker hay torchrun là nạn nhân cụ thể.

Recovery 6 giữ nguyên toàn bộ sửa RAM của version 5 và bổ sung:

- Mỗi tiến trình recovery đọc `memory.events` mỗi 10 giây. Nếu `oom_kill`
  hoặc `oom_group_kill` tăng so với lúc tiến trình bắt đầu, ghi
  `host_oom_detected`, dừng các tiến trình con đã theo dõi và thoát mã 86.
  OOM cũ của cgroup không khiến phiên mới bị dừng nhầm. Việc này phản ứng sau
  OOM, không bảo đảm tránh OOM hoặc cứu state chưa commit.
- Theo dõi PID và starttime qua `/proc`, bao gồm worker có session riêng và
  worker đã bị reparent sau khi launcher chết. Chỉ dừng tiến trình thuộc cây
  train, kiểm tra starttime để tránh kill PID đã được tái sử dụng. Đây là
  best effort: vẫn phụ thuộc supervisor còn sống và được lên lịch; không thể
  bảo đảm nếu cả môi trường notebook bị hệ thống kill.
- Chỉ launcher gốc in heartbeat mỗi 60 giây, gồm step hoàn tất, tổng step,
  trajectory đã xử lý/tổng, checkpoint đã xác nhận, pha từng rank, RAM cgroup
  và thời gian từ event rank cuối. Forward bắt đầu chưa được tính là step
  hoàn tất; checkpoint_started chưa được tính là backup thành công.
- `/kaggle/working/first-run-recovery/progress.log` giữ các dòng tiến độ gọn,
  `status.json` được thay nguyên tử bằng snapshot mới nhất mỗi phút.
  `memory-<pid>.jsonl`, rank logs và torchrun stdout/stderr vẫn đầy đủ.
  Heartbeat không phải bằng chứng tiến độ: nếu `last rank event` tăng liên
  tục và step đứng yên thì chỉ monitor còn sống. Chưa thêm timeout cho mọi
  kiểu treo không có OOM; không hứa console Kaggle luôn đồng bộ hoặc không cắt.

Kiểm thử dùng hai worker torchrun CPU thật, tín hiệu OOM được mô phỏng bằng
file cgroup, kiểm tra cả trường hợp launcher sống và launcher chết trước:
không còn worker sống sau khi watchdog thoát. Không gây OOM thật hoặc chạy
GPU để kiểm thử. Kiểm thử log phân biệt step/phase, backup đang ghi/đã hoàn
tất, chịu được JSONL đang ghi dở và giữ log file khi console được rút gọn.

**Lượt chạy tiếp theo dùng recovery dataset version 6**, không dùng version 5:
gắn version 6 trong Inputs của cùng notebook, giữ Secret và dataset epoch,
đổi `/versions/5` thành `/versions/6` trong cell ở trên, rồi native
Save Version → Save & Run All một lần. Nếu đã có lượt GPU đang chạy thì
code mới không tự áp dụng vào lượt đó; không tự mở thêm phiên song song.
Chế độ Draft có thể giữ GPU session sau khi cell lỗi, cần Stop Session để
thu hồi tài nguyên; watchdog dừng tiến trình train, không gọi API hủy Kaggle.

### Recovery 8: kiểm tra RAM trước batch, tự chọn song song hoặc lần lượt

Mục này ghi lại bản 8 đã xuất bản; bản thay thế là **Recovery 9** ở cuối tài liệu.

Phiên recovery 6 đã lưu/publish checkpoint 370 lúc `2026-10-17T21:40:57Z`
(04:40:57 ngày 18/10, UTC+7). Hai rank tiếp tục update 371 với 29.671 và
26.737 token. Pinned allocator đã về 0 trước forward nhưng RAM tiếp tục tăng
do activation đang dùng. Cgroup chạm giới hạn 30 GiB, `oom_kill=1`;
watchdog thoát mã 86 lúc `21:41:35Z`. Checkpoint 370 có đủ adapter, state hai
rank và manifest `complete=true` trong Output Kaggle. Payload trọng số của
bản này chưa được tải về/hash lại trên máy cá nhân trong lượt điều tra.

`cloud/kaggle/recovery/adaptive_memory.py` được cài qua `sitecustomize.py`
với `FIRST_RUN_ADAPTIVE_MEMORY=1`. Trước decoder forward, hai rank đã dọn cache
và trao đổi độ dài sample; rank 0 đọc giới hạn RAM cgroup, quyết định chế độ
rồi broadcast để mọi rank dùng cùng một lịch thực thi:

- Ước lượng theo size class power-of-two của pinned allocator, hidden width
  2.560 FP16. Envelope ban đầu là 36 block và 64 MiB overhead mỗi rank,
  nhân 1,15. Log recovery 6 quan sát được mức khoảng 4,5 GiB/rank tăng lên
  9 GiB khi input vượt khoảng 26.214 token. Không dùng tổng token đơn thuần.
- Baseline giữ anonymous memory, shmem, kernel, dirty/writeback; chỉ trừ
  phần file cache sạch có thể reclaim. `shmem` nằm trong `file`, không được
  trừ nhầm. Thiếu thống kê thì tính bảo thủ; thiếu giới hạn RAM thì từ chối.
- Dự phòng tối thiểu **3 GiB hoặc 10% giới hạn RAM**, lấy số lớn hơn. Nếu
  baseline + activation hai rank + dự phòng vừa RAM thì chọn `parallel`.
  Nếu chỉ baseline + activation của rank lớn nhất + dự phòng vừa thì chọn
  `serial`. Nếu cả serial không vừa thì báo lỗi trước forward, giữ checkpoint
  đã commit gần nhất; không cố ghi checkpoint trong lúc thiếu headroom.
- Với serial, rank 0 forward/backward rồi dọn pinned cache trước khi rank 1
  forward/backward. Nhánh này gọi cùng decoder module trong `DDP.no_sync`,
  tránh collective rebuild bucket ngầm trong `DDP.forward` gây deadlock.
  Khi cả hai hoàn tất, all-reduce scaled gradients SUM/2 trước unscale,
  clipping và **một** optimizer update. Batch lẻ vẫn dùng hệ số padding 2/0.
- Nhánh parallel giữ DDP forward/backward hiện tại. Calibration quan sát
  `allocated_bytes.current` trước cleanup; chỉ tăng envelope khi cần, không
  tự hạ ngưỡng sau một batch nhỏ. Không dùng counter `active_bytes.current`
  bất thường từ runtime này để suy ra dung lượng khả dụng.

Giữ nguyên hai sample trong mỗi batch, thứ tự seed 42, toàn bộ context, dtype,
LoRA r=32, loss và optimizer. Original archive/trainer/checkpoint identity vẫn
giữ nguyên; hash hook mới và lịch reduction được ghi trong recovery manifest.
Checkpoint format, state optimizer/scaler và RNG rank không đổi. Chỉ thứ tự
thực hiện và gradient reduction có thể khác, không hứa bit-exact CUDA.

Console ghi `batch_execution_mode` với step, mode, độ dài, baseline, estimates,
reserve và lý do. Full event vẫn vào file `memory-<pid>.jsonl`. Watchdog OOM và
cleanup trước checkpoint của recovery 6 được giữ. Estimator không bảo đảm
không OOM: RAM ngoại sinh, allocator hoặc một sample vượt VRAM vẫn có thể lỗi;
serial không tăng VRAM của từng GPU. Chưa có kết quả throughput hoặc GPU thực
tế của recovery 8. Không bắt OOM rồi tự retry vì kernel có thể SIGKILL worker.

Test hai process Gloo thật so sánh với DDP chuẩn qua các chuỗi mode đổi qua lại,
bao gồm serial ngay từ đầu/sau resume, dropout RNG, state optimizer/scaler,
batch cuối lẻ và việc cả hai rank từ chối khi serial không đủ headroom. Test
policy riêng bao gồm cặp 29.671/26.737 token gây lỗi và bước nhảy size class.
CPU không thay thế kiểm chứng PyTorch 2.11/CUDA T4.

Sau khi version 8 được publish/ready, dùng cùng notebook và Secret `HF_TOKEN`,
giữ dataset epoch, đổi cell bootstrap thành:

```python
import kagglehub, runpy
from pathlib import Path

bundle = Path(kagglehub.dataset_download(
    "cuonghoangquoc/first-run-memory-recovery-20261017/versions/8"))
runpy.run_path(str(bundle / "resume_first_run.py"), run_name="__main__")
```

Resume tải checkpoint latest đã kiểm hash. Nếu latest vẫn là 370, batch đầu là
371 và phải thấy event chọn serial với headroom tương tự lần điều tra. Phiên
GPU mới vẫn cần người dùng khởi chạy trong UI để giữ liên kết Secret; không
ghi lại notebook qua API, không chạy song song Draft và Saved Version.

### Recovery 9: bỏ DDP bucket, đồng bộ gradient thủ công

Lượt resume recovery 8 báo `CUDA OutOfMemoryError` tại MLP/LoRA dropout trong
backward: cần thêm 522 MiB, GPU 0 còn 468,81 MiB trống. Đây là lỗi VRAM, khác
host RAM OOM của recovery 6. Log r32/PagedAdamW32bit trước đó đã hoàn tất sample
29.974 token ở update 72 (cùng sample 24.013 token), peak allocated 13,82 GiB
trên rank 1; update 201 hoàn tất tổng 58.243 token (29.580/28.663). Không thể
quy lỗi chỉ cho độ dài hoặc suy rằng cấu hình mới chưa từng chạy gần 30k.

Recovery 8 giữ đối tượng DDP để đổi lịch, nhưng `no_sync()` chỉ tắt đồng bộ,
không giải phóng reducer/bucket. Nhánh serial có thể tạo gradient riêng; một
bộ gradient FP32 của 64.929.792 tham số chiếm 247,69 MiB/GPU. CPU characterization
xác nhận khác biệt chia sẻ storage; chưa có CUDA profiling để kết luận đây
là toàn bộ nguyên nhân OOM.

Recovery 9 thay cách thực thi, không chia MLP hoặc đổi cấu hình số học:

- Giữ hai model replica. Chặn đúng một lần constructor DDP trong lúc build,
  trả về module thường trước khi `DDP.__init__` chạy. Class DDP và các alias
  import vẫn là type hợp lệ; khôi phục `__new__` trong `finally`. Không tạo
  reducer/bucket, kể cả tạm thời. Frozen weights được load từ cùng revision đã
  kiểm hash; adapter được broadcast từ rank 0 sau khi kiểm signature hai rank.
- Giữ policy RAM và lịch parallel/serial của bản 8. Cả hai lịch đều all-reduce
  từng scaled gradient SUM/2 trước `GradScaler.unscale_`, clipping và một update.
  Rank 1 chỉ bắt đầu serial forward sau rank 0 backward và cleanup. Batch lẻ
  giữ đúng hệ số sample/padding 2/0.
- Giữ nguyên archive, trainer, index, identity config, rank32, dtype, optimizer,
  full context, thứ tự dữ liệu, checkpoint format, optimizer/scaler và RNG.
  Giữ các field DDP trong identity gốc để kiểm checkpoint; backend hiệu lực ghi
  riêng là `manual_data_parallel`. Thứ tự reduction có thể khác, không hứa
  bit-exact CUDA. Resume tải latest HF, không khởi tạo lại lượt train.
- Event `manual_replica_initialized` và `batch_execution_mode` ghi
  `ddp_bucket_bytes=0`. Watchdog của launcher kiểm tra rank log mỗi 10 giây;
  khi có event `failed` (kể cả CUDA OOM), ghi `rank_failure_detected`, dừng
  process train thuộc phiên và thoát 86, không đợi NCCL timeout. Không gọi API
  huỷ Kaggle; nếu Draft còn giữ GPU sau cell lỗi cần Stop Session trên UI.

Test hai process Gloo chạy cả hai thứ tự parallel/serial, nonreentrant checkpoint,
CPU saved inputs, dropout, odd batch, đồng bộ adapter lúc init từ hai seed khác
nhau, update/loss/optimizer/scaler/RNG so với DDP chuẩn và load checkpoint của
DDP chuẩn. Test cấm `DDP.__init__`, kiểm việc khôi phục constructor sau thành
công/thất bại, thiếu gradient, signature khác nhau và watchdog dừng peer khi
rank lỗi. Đây là bằng chứng CPU, không thay cho kiểm chứng T4/NCCL/PagedAdamW.
Không bảo đảm mọi batch vừa VRAM; vẫn cần quan sát batch 371 khi resume thật.

**Cập nhật Input trước khi Save & Run All:** trong cùng notebook, cập nhật
`first-run-memory-recovery-20261017` lên **dataset version 9**. Nếu không có nút
update thì gỡ riêng recovery Input và thêm lại bản mới. Giữ Input epoch và
Secret `HF_TOKEN`. Background session không thể tự attach version mới; đổi
chuỗi `versions/9` trong code khi Input vẫn v6/v8 sẽ lỗi trước train.

Cell khởi chạy đọc trực tiếp Input đã mount và từ chối nếu chưa đúng bản 9:

```python
from pathlib import Path
import hashlib, runpy

bundle = Path('/kaggle/input/datasets/cuonghoangquoc/first-run-memory-recovery-20261017')
expected = {
    'adaptive_memory.py': 'a9d7225aa7c7edffc84552568af5988f14a2d63b739744628d0c83145a25d174',
    'resume_first_run.py': '8f29982de9385037114b468918f1b2e96f8760e3b642a58087d6535e18788172',
}
for name, sha in expected.items():
    path = bundle / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise RuntimeError('Recovery Input chưa đúng version 9; cập nhật Input trước.')
runpy.run_path(str(bundle / 'resume_first_run.py'), run_name='__main__')
```

Chỉ khởi chạy một phiên bằng **Save Version → Save & Run All**, không chạy thêm
Draft song song. Nếu latest vẫn checkpoint 370, batch tiếp theo là 371. Khi
thấy `manual_replica_initialized` với `ddp_bucket_bytes=0`, backend mới đã cài;
chỉ event `step` 371 mới xác nhận update đó thành công.

### Recovery 10: lọc suffix chưa train dưới 28k

Người dùng chọn giữ context nguyên vẹn và bỏ bớt trajectory dài ở **phần chưa
train**. Mốc chuyển là checkpoint **375**, manifest SHA-256:
`af444f804ca64809bf2a8e6b7e0a3f7b94d10aad09b2d03de8f8f50ebad1a6e5`.
Update 376 đã chạy nhưng chưa checkpoint nên không được tính là tiến độ bền vững.

| Phần dữ liệu | Số trajectory |
| --- | ---: |
| Đã train, giữ nguyên prefix seed 42 | 750 |
| Chưa train tại mốc 375 | 10.777 |
| Chưa train, giữ lại vì `<28000` | 9.216 |
| Chưa train, bỏ vì `>=28000` | 1.561 |
| Toàn lịch continuation, gồm prefix | 9.966 |

Tổng lịch mới **210.967.717 token**, **4.983 update**, trong đó đã hoàn tất 375 và
còn **4.608 update**. Bước tiếp theo là **376**. Prefix vẫn có 105 trajectory
`>=28000` đã train; không lọc lại hoặc replay prefix. Chỉ số progress và mẫu số
trong log dùng lịch mới. Đây là một lượt qua tập đã lọc, không phải hoàn thành
đủ 11.527 trajectory của epoch gốc.

`cloud/kaggle/recovery/continuation-plan.json` cố định thứ tự source index:
750 hàng đầu của shuffle seed 42, sau đó các hàng suffix đủ điều kiện theo đúng
thứ tự tương đối cũ, ghép thành batch 2. Không shuffle lại, không cắt trajectory,
không đổi token/group/denominator, không ghi lại binary 1,2 GiB. `FilteredDataset`
là view dùng offset gốc; vẫn kiểm hash và tính toàn vẹn dataset nguồn.
Hash danh sách trajectory ID theo thứ tự:
`03fc2368bf4a287db13923845e929a75e0363ad32840008596cbc615fc257c69`.
Plan file SHA-256:
`42d0a70cec3f3688da0064524cc5a6e18919db1ac8f16299c37109648db261a3`.

Entrypoint mới import trainer gốc nguyên vẹn. Migration chỉ nhận đúng parent375
đã pin hash, kiểm identity số học/package/config và **mọi payload**, rồi load
adapter, optimizer, GradScaler, RNG từng rank và cursor bằng loader gốc. Identity
checkpoint mới bổ sung plan hash, ngưỡng, parent hash và implementation hash;
resume các lần sau dùng identity mới nghiêm ngặt. Plan không được tính lại từ
checkpoint latest. Sai plan/source/config/package phải dừng trước train; preflight
identity chạy trước model allocation.

Giữ backend manual data parallel, LoRA r32/alpha64, FP16 compute, FP32 adapter và
PagedAdamW32bit. Không dùng MLP chunking. Checkpoint mỗi 5 update, 1 local/3 HF,
cùng prefix `first-run` để không nhân đôi storage. Publisher cho phép duy nhất
**đúng parent375 → đúng toàn bộ child identity** dưới cùng prefix; các mismatch
khác bị từ chối. Commit chứa cả payload/manifest/latest, có parent-commit guard;
prune và squash giữ nguyên chính sách cũ. Sau resume từ375, save định kỳ tiếp là
380; dừng có kiểm soát có thể save sớm. Nếu chết trước save mới, resume lại375.

CPU tests kiểm thứ tự và biên `<28000`, tensor dữ liệu giữ nguyên, không replay
prefix, state model/AdamW/GradScaler/RNG khớp qua migrate→save→resume, payload lỗi
bị từ chối và fake HF kiểm parent republish→child publish→retention/retry/download.
Kiểm entrypoint với archive gốc và package metadata mô phỏng từ parent đã tới
parser thành công. Những kiểm tra này không chứng minh T4 hết OOM; không mở thêm
GPU pilot cho thay đổi bộ lọc.

**Cách chạy:** trong [cùng notebook](https://www.kaggle.com/code/cuonghoangquoc/workflow-aware-agent-dis-20261016-185315-746e15/edit),
cập nhật recovery **Input lên dataset version 10**, giữ Input epoch gốc và Secret
`HF_TOKEN` đang bật. Không sửa notebook qua API. Đổi code bootstrap khi Input còn
version cũ không cập nhật mount của phiên background. Dừng phiên cũ trước khi
**Save Version → Save & Run All** để chỉ có một writer checkpoint.

Cell đọc đúng Input đã mount, kiểm hash bootstrap của version 10:

```python
from pathlib import Path
import hashlib, runpy

bundle = Path('/kaggle/input/datasets/cuonghoangquoc/first-run-memory-recovery-20261017')
script = bundle / 'resume_first_run.py'
if hashlib.sha256(script.read_bytes()).hexdigest() != '84dc269946f1b5ba3f0c0d06b22742f7d3e48a524914772483b95d25625a6547':
    raise RuntimeError('Cập nhật recovery Input lên version 10 trước khi chạy.')
runpy.run_path(str(script), run_name='__main__')
```

Bootstrap phải báo `continuation_plan_verified` với `retained_prefix_count=750`,
`retained_suffix_count=9216`, `skipped_suffix_count=1561`, `total_records=9966`.
Nếu latest vẫn375, event `resumed` phải là375 và forward đầu376 có hai độ dài
**14.080/15.786 token**. Chỉ event `step`376 và `checkpoint_saved`380 với
`published=true` mới xác nhận update/backup mới trên GPU; không suy thành công
chỉ từ model loaded hoặc forward started. Lượt chuẩn bị/publish code chỉ dùng CPU.
