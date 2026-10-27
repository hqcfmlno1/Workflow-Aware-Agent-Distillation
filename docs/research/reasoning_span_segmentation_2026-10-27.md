# Cách chia và chấm reasoning span trong run LoRA hiện tại

Ngày kiểm tra: 2026-10-27. Branch `sft/offline-span-anchor`.
Code đang chạy: `src/workflow_distillation/lora_fp16/span_data.py` và
`span_cached_scoring.py`. Provenance profile: `coding-span-v2-section-anchor`.

## 1. Span là phần nào?

Một span là một khoảng ký tự liên tục **bên trong phần reasoning đã được render
của một assistant turn**. Một turn có thể chứa nhiều span. Không lấy cả trajectory
làm một span, không chia theo batch và không dùng số token cố định để chia.

Nguồn là `reasoning_content`, hoặc reasoning được normalizer trích từ `<think>`
trong content khi nguồn không có field riêng. Exporter định vị phần này bằng marker
qua **chat template chính thức**, rồi kiểm tra bỏ marker khôi phục đúng training text.
Token IDs và group masks phải trùng corpus vanilla đã pin.

Phần mở/đóng `<think>`, content, tool arguments, observation và EOS không thuộc
nội dung span dùng để ablate. Nếu chat template bỏ reasoning của một message,
exporter không chèn reasoning đó trở lại.

Các offset `[start, end)` là chỉ số ký tự Python trong **toàn bộ rendered text**,
không phải byte offset hoặc token index. Nội dung span là `text[start:end]`.

## 2. Quy tắc chia chính xác

Áp dụng riêng trong mỗi reasoning field, theo thứ tự sau:

1. Tạo ranh giới bằng regex:

   ```python
   r"(?:\n+|(?<=[.!?;:])[ \t]+)"
   ```

   Tách sau một hoặc nhiều newline, hoặc sau dấu `. ! ? ; :` khi dấu đó được theo
   bởi space/tab. Space/tab không đi sau các dấu trên không tạo ranh giới.
   Dấu câu, space/tab và newline khớp regex được giữ ở **cuối fragment trước**.
   Đây là quy tắc ký tự; không dùng parser câu tự nhiên hoặc Python AST.

2. Đếm độ dài fragment bằng số ký tự sau khi bỏ mọi whitespace theo `\s+`.
   Fragment **dưới 24 ký tự không trắng** được ghép vào fragment/span đứng trước,
   nếu đã có span trước. Ngưỡng là ký tự, không phải 24 token.

3. Nếu span đầu tiên vẫn dưới 24 ký tự và còn span tiếp theo, ghép nó với span sau.
   Lặp lại đến khi span đầu đủ dài hoặc toàn bộ reasoning chỉ còn một span.
   Vì vậy một reasoning ngắn vẫn có thể là một span dưới 24 ký tự.

4. Nếu còn hơn **24 span trong turn**, chọn cặp span kề nhau có tổng số ký tự
   không trắng nhỏ nhất và ghép lại. Tính lại rồi lặp đến khi còn 24 span.
   Khi bằng nhau, chọn cặp nằm bên trái trước. Không ghép qua ranh giới turn.

5. Giữ nguyên thứ tự và toàn bộ văn bản. Các span kề nhau phủ liên tục phần reasoning;
   nối tất cả span phải khôi phục đúng reasoning đã render, gồm cả whitespace.

Giới hạn 24 là **số span tối đa**, không phải độ dài span tối đa. Không có quy tắc
“mỗi span dài tối đa N token”; một span có thể dài khi câu không có dấu tách hoặc
khi nhiều fragment được gộp để đạt giới hạn 24.

Ví dụ tự tạo, chạy bằng đúng splitter:

```text
Plan:
Read the failing test and client implementation.
OK.
Then update the argument handling and run the targeted test.
```

Kết quả là hai span:

```python
[
    "Plan:\nRead the failing test and client implementation.\nOK.\n",
    "Then update the argument handling and run the targeted test.",
]
```

`OK.` ngắn nên được ghép ngược vào đoạn trước; `Plan:` ngắn ở đầu nên được ghép
tiếp vào span sau. Không bỏ hai fragment ngắn này khỏi dữ liệu.

## 3. Khi nào reasoning được chấm?

| Trạng thái | Xử lý |
|---|---|
| Có rendered reasoning và tool call: `eligible` | Chấm các span theo action cùng turn |
| Không có tool call: `no_action_unscored` | Không ablate; reasoning giữ CE weight 1 |
| Có tool call nhưng không có rendered reasoning: `no_rendered_reasoning` | Không ablate |
| Eligible nhưng chỉ có một span | Weight luôn 1 theo upstream; cached scorer bỏ forward chẩn đoán |

Nếu một turn chứa nhiều tool call, target là **một block chung** từ call đầu đến
call cuối, gồm separator ở giữa. Không chấm từng call độc lập. Submission vẫn có
group train riêng; block submit cũng có thể là action để chấm reasoning trước nó.

## 4. Xoá span và tính điểm

Với lịch sử gốc $h$, reasoning $r$, phần content/protocol còn lại $c$ và action
teacher $a$, đóng góp của span $s_i$ là:

$$
\Delta_i = \log P_{\mathrm{ref}}(a\mid h,r,c)
           -\log P_{\mathrm{ref}}(a\mid h,r\setminus s_i,c).
$$

Hai vế đều là tổng log-prob teacher-forcing của **cùng action token IDs**, dùng
frozen base Qwen3.5-4B. Không phải xác suất của reasoning, không gọi teacher model
và không thực thi tool trong scoring.

Chỉ xoá đúng ký tự trong span hiện tại. Lịch sử các turn trước, reasoning còn lại,
content/protocol và action teacher giữ nguyên. Không thay span bằng một lời nhắc,
không sinh reasoning mới và không cắt history. Mỗi biến thể được tokenize lại;
nếu action target tokenization thay đổi, scorer báo lỗi.

Scorer cached giữ một cache của lịch sử teacher nguyên bản và deep-copy toàn bộ
KV/convolution/recurrent state cho từng nhánh ablation. Nhánh đã xoá span không
được dùng làm lịch sử của turn tiếp theo. Nếu prefix token IDs không còn khớp,
scorer dựng lại cache thay vì tái sử dụng trạng thái khác lịch sử.

## 5. Từ đóng góp sang weight

Xếp hạng $\Delta_i$ **trong cùng turn** theo chiều tăng; chỉ các score bằng nhau
chính xác mới chia sẻ average rank. Với $n>1$, chuẩn hoá rank thành $u_i\in[0,1]$:

$$
\sigma=\max_j\Delta_j-\min_j\Delta_j,\qquad
\lambda=\min\left(1,\frac{\sigma}{0.05}\right).
$$

$$
w_i=0.1+0.9\left[\frac12+
\left(u_i-\frac12\right)\lambda\right].
$$

Weight được chặn trong $[0.1,1]$. Nếu mọi score bằng nhau, weight là 0.55.
Trường hợp $n=1$ là ngoại lệ riêng: weight 1. Đóng góp âm vẫn tham gia ranking,
không bị mask về 0; span có đóng góp âm vẫn có thể đứng đầu nếu các span khác âm hơn.

**Ngưỡng 0.05 nats áp dụng cho spread của cả turn.** Nó không làm mọi cặp score
cách nhau dưới 0.05 trở thành tie. Khi spread đã lớn hơn ngưỡng, các span gần nhau
vẫn có thể đổi thứ hạng vì sai số số học, kéo theo weight thay đổi đáng kể.

## 6. Ánh xạ về token train

Training vẫn dùng **sequence teacher nguyên vẹn**; sequence đã ablate chỉ dùng để
chấm điểm offline. Với token có character interval $[a,b)$, weight mặc định là 1:

$$
w_{\mathrm{token}}=1+
\sum_i\frac{|[a,b)\cap s_i|}{b-a}(w_i-1).
$$

Token nằm trong một span nhận weight của span đó. Token giao hai span nhận
trung bình theo số ký tự giao. Token offset rỗng giữ weight mặc định.
Sau phép ánh xạ, code chỉ giữ adaptive CE weight cho token có **group 1 (reasoning)**;
content, action, submit và protocol được supervise đều giữ CE weight 1.
User/tool/context/padding vẫn theo mask gốc và không tham gia loss.

Mask của anchor là một mảng độc lập. Closing `</think>` được supervise có anchor
weight cao theo cấu hình, nhưng không bị xoá như một reasoning span và CE weight
của nó vẫn là 1. Token mơ hồ giữa các field bị kiểm tra trong bước annotate;
whitespace giao với supplied prefix được mask theo quy tắc của corpus gốc.

## 7. Những giới hạn cần phân biệt với lỗi triển khai

- Splitter có thể cắt ở newline bên trong code block hoặc sau viết tắt/dấu hai chấm.
  Nó không bảo đảm mỗi span là một ý reasoning hoàn chỉnh, cũng không nhận diện lặp.
- Một span dài chưa chắc quan trọng; một span ngắn chưa chắc vô ích. Điểm đo khả năng
  dự đoán action teacher, không xác nhận reasoning đúng hoặc action sẽ giải được task.
- Hiệu ứng tương tác, các span trùng ý và độ dài action có thể ảnh hưởng đóng góp.
  Ranking là tương đối trong turn, không phải thang chất lượng tuyệt đối giữa trajectory.
- Full và cached BF16 không cho kết quả số học giống hệt. Đã đo một turn 35.690 token:
  sai khác mean action log-prob tối đa khoảng 0.00604 nats, weight lệch tối đa 0.50870.
  Sau khi biết số đo này, người dùng chọn chấm lại toàn bộ bằng cache và lưu riêng.
  Không trộn span score của scorer full cũ vào profile cached.
- Resume lưu điểm các biến thể đã xong, không serialize trạng thái cache GPU.
  Việc dựng lại history có thể thay đổi ranh giới chunk; chưa có audit riêng bảo đảm
  tái lập bitwise cho turn bị ngắt giữa các biến thể. Không suy diễn kết quả kiểm tra
  dữ liệu dưới đây thành bảo đảm đồng nhất số học trên mọi lịch sử stop/resume.

## 8. Nguồn và kiểm chứng

Quy tắc chia/gộp và rank/dead-zone được đối chiếu với
[`src/webkd/action_informativeness.py`](https://github.com/sparklabutah/webKD/blob/95cdc1d21ce3b590063f6ab9c5bbd3eb25063de5/src/webkd/action_informativeness.py),
revision `95cdc1d21ce3b590063f6ab9c5bbd3eb25063de5`.
Upstream xét các rationale tag `think/plan/memory`; adaptation này xét reasoning
field đã render của coding harness. Không áp dụng cấu trúc nhiều tag của web agent
lên tool JSON hay observation.

SHA256 nguồn upstream đã tải lại và đối chiếu:
`2085c4c6173b8e408abb8d67871556d588723df8b6a2a3e7af7dd86ac0d37bd5`.
SHA256 provenance index đang dùng trên server:
`129d8603b505bc6131a26bf53d9f0ded0490091e787e72931ce0206d3510798c`.

Audit mới kiểm tra **toàn bộ 1.180 record**, không chỉ lấy mẫu:

| Chỉ số của provenance hiện tại | Giá trị |
|---|---:|
| Turn có reasoning và action để chấm | 30.978 |
| Tổng span trong các turn được chấm | 157.298 |
| Trung bình span/turn được chấm | 5,08 |
| Turn chỉ có một span, weight cố định 1 | 12.013 |
| Turn thực sự cần gộp vì vượt 24 span | 1.748 |
| Turn có đúng 24 span sau xử lý | 1.848 |
| Số span trước giới hạn lớn nhất trong một turn | 193 |
| Token giao với một span: median / p90 / p99 / max | 20 / 44 / 99 / 335 |
| Ký tự trong một span: median / max | 79 / 967 |
| Turn được chấm chứa nhiều tool call | 5.252 |
| Turn được chấm dùng block submit làm action | 1.163 |

“Token giao với span” đếm token có offset chạm span, không tokenize riêng span;
token giao hai span có thể được đếm trong cả hai thống kê. Giá trị 335 là max đo
được trong corpus này, không phải một token cap trong code.

30 record có tổng cộng 400 message bị template bỏ reasoning; exporter không phục
hồi chúng. Tổng `no_rendered_reasoning` là 993 turn, bao gồm cả reasoning rỗng;
không diễn giải toàn bộ 993 turn là bị template xoá.

Các kiểm tra đã qua:

1. Hash của corpus, từng provenance file, tokenizer/template; code local khớp
   code trên Vast đang chạy.
2. Token IDs và offset mapping của mọi rendered record khớp corpus.
3. Tất cả span liên tục, có độ dài dương, nằm trong đúng reasoning field; không
   lấn vào action hoặc closing tag; phân hoạch khớp splitter hiện tại và upstream.
4. Tokenization của mọi original action prefix và target khớp dữ liệu export.
5. 33 test CPU liên quan đến data/scoring/cache/loss/resume đều qua.
6. Snapshot **148 record đã chấm xong**, gồm 4.100 turn và 944.315 supervised token:
   tính lại token weights bằng một cách vector hoá độc lập, sai khác tối đa **0**.
   Reference log-prob và anchor mask khớp hoàn toàn các mảng nguồn.

Không phát hiện lỗi chia span, trộn field hoặc ánh xạ token trong phạm vi kiểm tra
này. Các giới hạn phương pháp và số học ở mục 7 vẫn còn. Kiểm tra không thay splitter,
công thức, corpus hoặc job scoring đang chạy và không chạy thêm forward GPU.

Số đo/hashes được lưu trong
[báo cáo audit JSON](reasoning_span_audit_2026-10-27.json).
