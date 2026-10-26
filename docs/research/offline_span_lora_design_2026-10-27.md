# LoRA với offline action-informed reasoning weights và frozen-base anchor

Trạng thái: đề xuất thí nghiệm, chưa triển khai hoặc chạy GPU. Ngày 2026-10-27.

## Mục tiêu và phạm vi

Kiểm tra liệu giảm học những đoạn reasoning ít giúp dự đoán action, đồng thời
hạn chế student lệch khỏi base trên reasoning, có cải thiện hành vi so với vanilla
SFT hay không. Không mặc định rằng lặp reasoning là do undertrain hoặc học văn phong.

Lấy cảm hứng từ [WebKD, arXiv:2610.19834](https://arxiv.org/abs/2610.19834).
Đối chiếu chi tiết trong [reading notes](webkd_2610_19834_2026-10-27.md).
Đây là adaptation của weighting và anchor, không phải tái lập toàn bộ WebKD:
không BC seed, không student rollout để lấy failure states, không teacher editing.
Reference của paper là BC seed; reference ở đây là base trước SFT. Phương pháp này
chưa xử lý trực tiếp exposure bias trên các state do student tự tạo.

## Các yếu tố giữ nguyên để đối chiếu

| Thành phần | Đề xuất |
|---|---|
| Base/reference | `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Khởi tạo | LoRA mới từ base, không resume adapter 840/107 hoặc run bị suy giảm |
| Dtype | BF16 base, không quantize; giữ FP32 LoRA/optimizer như trainer hiện tại |
| LoRA | rank 32, alpha 64, dropout 0.05; giữ nguyên target modules của baseline |
| Dữ liệu | 1.180 successful attempts DeepSeek-v4-Pro, 477 task, tối đa 35.998 token |
| Selection | Giữ toàn bộ selection hiện tại; TQS đã sort pool, không thêm cutoff/top-K |
| Epoch/batch | 2 epoch; microbatch 1 trajectory, accumulation 4; 295 update/epoch |
| Optimizer | LR 1e-4, cosine, warmup 0.05, weight decay 0.01, clip 1.0 |
| Memory | Gradient checkpointing, FLA, SDPA, hybrid activation storage như baseline |
| Holdout | Không tự tách thêm: baseline hiện tại không holdout |

LR được giữ để cô lập thay đổi objective; đây không phải khẳng định LR tối ưu.
Dataset index SHA256: `25ba49de9761df40cc9ed0ecb8568a8e02a71f9c59fc1029303e968d604eeebc`.
Nguồn cấu hình: `configs/training/gpus/rtx5090_32gb/metacurate_bf16.json`.

Giữ nguyên token IDs, masks và thứ tự shuffle của corpus vanilla. Audit hiện tại
có 30 trajectory bị official template bỏ reasoning ở một số message. Chỉ reasoning
thực sự còn trong rendered training sequence mới được chấm; không tự chèn reasoning
bị bỏ trở lại. Thay renderer cần một matched vanilla control riêng.

## Giai đoạn A: offline đúng một lần

Reference chạy `eval`, tắt dropout, không adapter đã train. Hai đầu ra độc lập:

1. `ref_logp`: một float32 cho mỗi supervised target token trên **nguyên sequence**.
2. `reasoning_weight`: trọng số cho token reasoning, dựa trên span ablation.

Với turn có history $h$, reasoning $r$ và action $a$, đặt:

$$
\Delta_s=\log P_{\rm ref}(a\mid h,r)
         -\log P_{\rm ref}(a\mid h,r\setminus s).
$$

Các content/protocol token nằm giữa reasoning và action được giữ nguyên trong cả
hai điều kiện, viết tắt khỏi công thức. Dùng tổng log-prob của cùng action token
sequence, teacher forcing; không sinh action, không thực thi tool. Nếu turn có nhiều
tool calls, action là toàn bộ block từ call đầu đến call cuối, gồm separator ở giữa.
Submit vẫn là group riêng, cũng có thể là action dùng để score reasoning trước nó.

Tách reasoning theo câu/newline, gộp fragment dưới 24 ký tự không trắng, tối đa
24 span/turn bằng gộp cặp span kề nhau ngắn nhất; không cắt bỏ text. Tie khi chọn
cặp thì chọn cặp bên trái. Dùng sentence/newline splitter và rank semantics của
public WebKD code làm hợp đồng, pin revision khi triển khai.

Với average rank chuẩn hóa $u_s\in[0,1]$, đặt độ chênh lệch score:

$$
\sigma=\max_s\Delta_s-\min_s\Delta_s.
$$

$$
v_s=\operatorname{clip}_{[\phi,1]}\left\{
\phi+(1-\phi)\left[\tfrac12+(u_s-\tfrac12)
\min(1,\sigma/\tau)\right]\right\},\quad
\phi=0.1,\quad\tau=0.05\ {\rm nats}.
$$

Đối chiếu upstream revision `95cdc1d21ce3b590063f6ab9c5bbd3eb25063de5`:
turn chỉ có một span nhận $v_s=1$; nhiều span có cùng score nhận $v_s=0.55$.
Trường hợp singleton là special case của code, đính chính mô tả midpoint trong draft. Không hard-mask $\Delta_s$ âm.
Đây là relative ranking trong turn; score cao không chứng minh reasoning đúng.
Nếu turn không có action, reasoning giữ weight 1 và được ghi `no_action_unscored`;
không gán score giả. Turn không có rendered reasoning không cần ablation.

Map $v_s$ xuống token bằng trung bình theo character overlap. Chỉ nội dung reasoning
nhận adaptive weight; content, action, submit, delimiter/EOS được supervise giữ 1.
Context/padding/supplied assistant prefix giữ mask cũ và không tham gia loss.
Token crossing giữa hai span reasoning lấy overlap mean; crossing field/protocol
không rõ ràng phải fail để kiểm tra, không đoán nhãn.

Chấm trên exact rendered prefix. Sau khi xóa span và tokenize lại prefix, nối lại
**cùng action target IDs** tại protocol boundary đã xác minh; bản full phải khôi phục
đúng IDs của corpus. Không rebuild history bằng template làm thay đổi mask/omission.
Cache reference dùng prefix gốc, tuyệt đối không dùng prefix đã ablate.

Lưu sidecar theo corpus offsets cùng manifest: model/revision, tokenizer/template
hash, corpus hash, scoring dtype/backend, splitter version, config, completion status.
Scoring resume theo trajectory/turn và ghi atomic. NaN/Inf hoặc alignment lỗi là lỗi
chưa hoàn thành; không tự fill weight mặc định rồi cho train.

## Giai đoạn B: loss khi train LoRA

Với một trajectory $x$, gọi $S_x$ là tập token được supervise và $x_{<i}$
là prefix gốc trước token thứ $i$. Đặt token loss:

$$
\ell_i=-\log p_\theta(y_i\mid x_{<i}).
$$

Reference log-prob được tính offline một lần và giữ cố định:

$$
q_i=\log p_{\mathrm{ref}}(y_i\mid x_{<i}).
$$

$$
\mathcal L_{\rm CE}(x)=
\frac{\sum_{i\in S_x}w_i\ell_i}{\sum_{i\in S_x}w_i}.
$$

Trọng số CE cho từng token:

$$
w_i=\begin{cases}
v_i & \text{token nội dung reasoning},\\
1 & \text{các target khác.}
\end{cases}
$$

$$
d_i=\operatorname{clip}\left(q_i-\log p_\theta(y_i\mid x_{<i}),-20,8\right),
\quad k_i=\exp(d_i)-d_i-1.
$$

$$
\mathcal L_{\rm anchor}(x)=
\frac{\sum_{i\in S_x}\gamma_i k_i}{\sum_{i\in S_x}\gamma_i}.
$$

Loss cuối cùng:

$$
\boxed{\mathcal L(x)=\mathcal L_{\rm CE}(x)+\beta\mathcal L_{\rm anchor}(x)},
\qquad \beta=0.03.
$$

Theo rationale mask trong implementation WebKD, dùng role weights:

$$
\gamma_i=\begin{cases}
1 & \text{nội dung rationale hoặc response-envelope token},\\
10^{-6} & \text{các supervised response token khác.}
\end{cases}
$$

Khi chuyển sang coding, reasoning là rationale; content, action và submit có
anchor weight $10^{-6}$, còn response-envelope tokens giữ $1$. Cần map envelope
theo template và lưu mask riêng, không suy ra từ group action/protocol hiện tại.
Token context/padding vẫn bị loại khỏi cả hai loss. Không có reasoning không đồng
nghĩa anchor bằng 0 vì envelope và epsilon support vẫn tồn tại. Chỉ khi không có
anchor support mới đặt anchor bằng 0. Đây là sửa đổi ngày 2026-10-27, thay cho đề xuất
trước đó đặt non-reasoning bằng 0; không chuyển sang uniform all-token anchor.

Đây là role mask cố định, không phải adaptive weight được tính từ độ lệch phân phối.
Tool/action gần như không bị anchor để tự do học corrected decisions. Reasoning
anchor không nhân thêm $v_i$: adaptive CE và anchor là hai cơ chế khác nhau.
$\beta=0.03$ là điểm xuất phát từ paper,
không phải giá trị đã được tối ưu trên coding.

Mean loss của từng trajectory rồi mean qua 4 trajectory/update, giữ reduction của
baseline. Không chuyển ngầm sang per-turn records hoặc global token mean cả batch.
Không dùng lại hybrid group-length normalization hay fixed reasoning weight $2$ của run cũ.

Anchor là **target-token k3 penalty trên corpus cố định**, không phải full-vocabulary
KL, không phải unbiased KL estimate trên offline teacher tokens. Nó ràng buộc drift
ở những prefix/token đã quan sát, không bảo đảm giữ toàn bộ coding capability.
Ở điểm current=reference với cùng evaluation mode, anchor và gradient bằng 0;
dropout khi train có thể làm anchor ban đầu khác 0.

Reference log-prob cache cho phép train chỉ nạp một base + LoRA lên GPU. Tính
current log-prob từ existing chunked output-head CE; anchor/exp/reduction FP32,
dùng `expm1(d)-d` để ổn định gần 0. Không materialize seq-length × vocabulary logits.
7.761.483 supervised tokens hiện tại cần khoảng 31 MB decimal cho ref_logp float32,
thêm khoảng 31 MB nếu lưu weight float32 riêng, chưa tính metadata/ablation logs.

## Giai đoạn C: triển khai, log và đánh giá

- Trước toàn pool: đo scoring trên vài turn thật có history ngắn/vừa/dài và đếm
  số eligible turns/spans toàn corpus bằng CPU. Ước lượng sum(full+ablations),
  không suy ra chi phí từ 1.180 trajectory. Tối đa 25 scoring variants/eligible turn;
  latency còn phụ thuộc context. Không cam kết toàn pipeline vừa ngân sách 3 USD.
- Dùng cache/prefix reuse khi đã xác minh tương đương; không cần hai model resident.
  Sau scoring unload scorer rồi load train. Chỉ kiểm tra GPU một trajectory dài nhất
  forward/backward trước epoch; không lặp nhiều pilot/rollout không cần thiết.
- Log mỗi optimizer update: total, weighted CE, plain CE, anchor, LR, grad norm,
  per-field plain NLL/count, sum weights, reasoning weight quantiles/effective mass,
  throughput và peak VRAM. Plain CE hữu ích đối chiếu; total loss giữa objectives
  khác nhau không so trực tiếp để kết luận student tốt hơn.
- Giữ một local recovery mới nhất và checkpoint cuối từng epoch. Upload HF chỉ
  cuối epoch kèm config, manifest/hash, `events.jsonl`, `train.log`; retry upload
  không xóa checkpoint local chưa upload thành công. Không lưu/copy toàn base mỗi lần.
- Không gọi train CE là validation loss. Chưa có holdout thì không có independent
  validation curve. Có thể dùng một tập development task ngoài 477 task train,
  task-disjoint và kiểm tra contamination, nếu bổ sung sau.
- So sánh vanilla, weighted CE-only ($\beta=0$) và weighted CE+anchor khi đủ ngân sách;
  nếu chỉ một run mới, ưu tiên combined đã thảo luận nhưng không quy công cho riêng
  weights hay anchor. Dùng cùng harness, decoding, task set và checkpoint epoch.
- Báo cáo: submit/patch, test resolved, reasoning loop, tool repetition, token/turn
  cost, đọc lại observation và sửa giả thuyết. Tách infrastructure failures khỏi
  model failures. Task debug cũ chỉ là case study, không thay thế evaluation độc lập.

## Điều kiện kết thúc chuẩn bị

Đủ cache/alignment kiểm chứng, estimate chi phí thực tế, config mới riêng và một
entrypoint resume được. Tài liệu này không cấp lệnh thuê GPU, gọi teacher API,
train hoặc publish dữ liệu. Không sửa artifacts của những run lịch sử.
