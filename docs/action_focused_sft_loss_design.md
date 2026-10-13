# Action-focused SFT — Thiết kế hàm loss Phase 1

**Qwen3.5-4B · mini-SWE-agent · baseline 4.000 trajectory**
Ngày: 13/10/2026 · Phiên bản thiết kế: 1.0

## 1. Mục tiêu và quyết định thiết kế

Phase 1 tinh chỉnh một student đã có năng lực agent, ưu tiên giúp student tuân thủ protocol mini-SWE-agent, sinh tool calls đúng cấu trúc, sử dụng Bash và kết thúc task đúng cách. Giữ reasoning và lời giải thích của teacher trong hội thoại, đồng thời giảm mức ưu tiên của chúng trong objective. Không xóa reasoning để biến dữ liệu thinking thành non-thinking.

Thiết kế sử dụng **cross-entropy với nhãn cứng của teacher**, chuẩn hóa riêng theo bốn nhóm token: reasoning, content, action và submit. Đây là SFT lấy cảm hứng từ segment-weighted supervision của SmartAD; không phải KL distillation vì không có phân phối logits của teacher.

Các quyết định của baseline:

- System, user, tool observations và prefix do harness cung cấp: chỉ làm context, không tính loss.
- Reasoning/content do assistant sinh: có loss, với hệ số thấp.
- Tool calls thông thường và các delimiter cần thiết do model sinh: có loss cao.
- Tool call submit kết thúc trajectory: một nhóm riêng, không tính lại trong action.
- Chuẩn hóa loss theo số token trong từng nhóm, sau đó trung bình theo trajectory/task.
- Hệ số khởi đầu: reasoning 0,10; content 0,05; action 1,00; submit 0,25.

Các hệ số là lựa chọn thiết kế cần kiểm chứng trên validation, không phải kết quả thực nghiệm hoặc giá trị tối ưu đã được xác nhận. Công thức không phụ thuộc việc base weights được lưu theo LoRA 16-bit hay QLoRA 4-bit.

Giữ target reasoning với loss khác 0 giúp tiếp tục supervise cách sinh reasoning, nhưng không bảo đảm năng lực thinking được giữ nguyên. Training và validation phải ghi rõ thinking mode/prefix đang dùng; không đổi sang non-thinking trong khi gọi đây là cùng một baseline.

## 2. “Final decision” được định nghĩa như thế nào?

SmartAD dùng block hành động cuối tạo đáp án làm vùng final. Trong mini-SWE-agent, bản sửa được tạo dần qua các action sửa code; action submit chỉ chọn thời điểm kết thúc và chuyển kết quả cho harness. Vì vậy, tài liệu này dùng tên **submit/termination**, thay vì coi submit tương đương hoàn toàn với đáp án cuối của bài QA/math.

Trong cấu hình SWE-bench của mini-SWE-agent v2.4.6, user prompt yêu cầu:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

Harness dựng prompt từ `instance_template` ngay đầu mỗi task. Docker environment nhận submit khi dòng đầu output là tín hiệu trên và return code bằng 0, rồi lấy phần output còn lại làm submission. Harness không tự xác định patch đã giải đúng bài và không ép student submit tại một thời điểm do harness chọn.

Audit ngày 13/10/2026 trên 49.925 trajectory thành công đã tải xác nhận: tất cả có hướng dẫn submit trong prompt ban đầu, tất cả kết thúc bằng một Bash tool call phát tín hiệu submit, với 23 biến thể command. Trong đó 35.586 dùng nguyên văn command; 11.828 thêm `cd /testbed &&`; 2.473 dùng đường dẫn `/testbed/patch.txt`; 38 dùng biến thể khác.

Không nâng trọng số của toàn bộ assistant message cuối. Reasoning và bản tóm tắt dài trong message đó vẫn thuộc các nhóm reasoning/content. Việc nộp thành công về protocol không chứng minh patch đúng; `resolved == 1` là kết quả được ghi bởi nguồn dataset.

<!-- pagebreak -->

## 3. Phân vùng token và tạo supervision

Gọi G = {R, C, A, F}, lần lượt là reasoning, content, action và final submit. Mỗi token được supervise thuộc đúng một nhóm; token context không thuộc G.

| Nhóm | Phạm vi | Hệ số λ | Quy tắc |
| --- | --- | --- | --- |
| Context | System, đề bài, tools schema, observations, assistant prefix đã cung cấp | 0 | `labels = -100`; vẫn giữ trong input |
| R — Reasoning | Nội dung suy luận assistant được template thực sự serialize | 0,10 | Không tự lấy lời giải thích trong observation làm reasoning |
| C — Content | Lời nói assistant ngoài reasoning và ngoài tool calls | 0,05 | Bao gồm bản tóm tắt trước submit |
| A — Action | Các tool calls không phải terminal submit; protocol delimiters do model phải sinh | 1,00 | Gồm tên function, tham số, toàn bộ command và nội dung code sửa file |
| F — Submit | Tool call thực hiện submit cuối cùng của trajectory | 0,25 | Gồm wrapper, tên function, tham số và command; loại khỏi A |

**Cấu trúc tool call của student.** Dữ liệu normalized có `tool_calls[].function.name` và `function.arguments`; template chính thức Qwen3.5 serialize thành cấu trúc dạng sau. Đây là minh họa, không phải template tự viết thay thế tokenizer:

```text
<tool_call>
<function=bash>
<parameter=command>
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
</parameter>
</function>
</tool_call>
```

**Quy tắc phân vùng bắt buộc:**

1. Dùng role, các trường structured và vị trí render đã đối chiếu để xác định vùng. Không tìm marker submit hay code fence trên toàn bộ text rồi gán loss; marker xuất hiện cả trong prompt và reasoning.
2. F chỉ áp dụng cho tool call submit cuối thực sự kết thúc trajectory. Một attempt submit trước đó thất bại vẫn là action A; không gán mọi command chứa marker thành F.
3. Nếu message có nhiều calls, phân loại từng call. Một call F không biến các calls khác hay toàn bộ message thành F.
4. Trường hợp kết thúc không có observation cuối được cho phép theo dữ liệu đã audit. Không tạo observation hoặc đáp án patch mới để bù vào.
5. Giữ command nhiều dòng, Unicode, quotes và escapes theo dữ liệu normalized. Parse đúng một lần ở mỗi lớp JSON; không sửa command để làm nó dễ học hơn.
6. Prefix assistant mà inference đã cung cấp phải mask. Delimiter chuyển pha do model phải sinh, như `</think>` và delimiter ngoài tool calls, thuộc A. Delimiter kết thúc message sau submit thuộc F; nếu sau action thường thì thuộc A. Không supervise một token kết thúc vốn đã nằm trong input prefix.
7. Token newline/whitespace thuộc vùng render chứa nó. Delimiter bên trong tool-call wrapper kế thừa A hoặc F của call đó. Mọi token assistant còn lại phải có nhãn xác định, không rơi vào nhóm mặc định âm thầm.

Patch xuất hiện trong tool observation chỉ là context. Code được assistant viết trong command EDIT là target A. Không đưa reference patch vào input hoặc vào target mới.

## 4. Template, reasoning và ranh giới token

Dùng tokenizer `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; checksum chat template `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`. Dữ liệu hiện tại render với `add_generation_prompt=False`.

Template đã lưu có điều kiện theo user query cuối: reasoning của assistant sau query đó được render; reasoning ở phần lịch sử trước query mới có thể bị bỏ. Không mặc định reasoning chỉ tồn tại ở response cuối, cũng không mặc định mọi `reasoning_content` normalized đều có token tương ứng trong text.

Chỉ tính loss trên token thực sự đi vào model. Khi triển khai phải báo cáo reasoning có trong normalized nhưng không được render. Nếu thay đổi cách dựng sample để supervise lại reasoning bị bỏ, đó là thay đổi representation cần version và kiểm chứng riêng; không âm thầm sửa template.

Tokenize toàn bộ text bằng tokenizer chính thức; không tokenize từng vùng rồi nối vì kết quả có thể khác. Chuyển character spans thành token spans bằng offset mappings và xử lý riêng special tokens. Ranh giới không rõ, token cắt qua hai vùng hoặc không đối chiếu được phải được ghi lỗi và xử lý trước training; không tự gán thành action. Các `action_candidates` hiện có mới là character spans ứng viên, chưa phải token labels hoàn chỉnh.

<!-- pagebreak -->

## 5. Công thức loss chính thức của baseline

Với trajectory n, gọi yₙ,ᵢ là token target ở vị trí i; hₙ,<ᵢ là toàn bộ context đứng trước nó. Cross-entropy ở token đó:

$$\ell_{n,i}(\theta)=-\log p_\theta(y_{n,i}\mid h_{n,<i}).$$

Gọi Iₙ,ᵍ là tập vị trí target thuộc nhóm g; Nₙ,ᵍ = |Iₙ,ᵍ|. Tính mean loss riêng cho mỗi nhóm không rỗng:

$$\bar L_{n,g}=\frac{1}{N_{n,g}}\sum_{i\in I_{n,g}}\ell_{n,i},\qquad N_{n,g}>0.$$

Gọi Gₙ là các nhóm có Nₙ,ᵍ > 0. Loss của trajectory:

$$L_n=\frac{\sum_{g\in G_n}\lambda_g\bar L_{n,g}}{\sum_{g\in G_n}\lambda_g}.$$

Loss của một tập/batch chứa B trajectory, mỗi trajectory có trọng số ngang nhau:

$$L_{\mathrm{phase1}}=\frac{1}{B}\sum_{n=1}^{B}L_n.$$

Với đầy đủ bốn nhóm, công thức cụ thể là:

$$L_n=\frac{0.10\bar L_{n,R}+0.05\bar L_{n,C}+1.00\bar L_{n,A}+0.25\bar L_{n,F}}{1.40}.$$

Nếu một nhóm rỗng, bỏ cả tử và hệ số của nhóm đó khỏi mẫu số; không chia cho 0 và không tự thêm target. Trajectory không có A hoặc F trong baseline hiện tại phải bị báo lỗi dữ liệu thay vì âm thầm tiếp tục. Nhóm R rỗng cần được đối chiếu với reasoning normalized và điều kiện của template trước khi chấp nhận.

### 5.1. Ý nghĩa của hệ số

| Nhóm | Tỷ lệ hệ số trong objective khi đủ bốn nhóm |
| --- | --- |
| Reasoning | 7,14% |
| Content | 3,57% |
| Action | 71,43% |
| Submit | 17,86% |

Đây là tỷ lệ hệ số của các mean losses, **không phải tỷ lệ gradient thực tế, tỷ lệ VRAM hoặc tỷ lệ thời gian train**. Gradient còn phụ thuộc độ khó và giá trị loss của từng token.

Reasoning dài không tự nhận thêm tỷ trọng chỉ vì có nhiều token. Submit có hệ số nhỏ hơn tổng action nhưng lớn hơn reasoning/content. Vì submit thường rất ngắn, hệ số thực trên từng token F có thể lớn hơn từng token A: với cùng một trajectory, hệ số token nhóm g bằng λᵍ / (Zₙ × Nₙ,ᵍ), trong đó Zₙ = ∑ᵍ∈Gₙ λᵍ. Không cần đặt λF cao nhất mới tạo được ưu tiên mạnh cho submit.

Không dùng λF = 2 làm mặc định khi đã chuẩn hóa theo nhóm: khi đó một câu submit ngắn sẽ nhận nhiều hệ số hơn tất cả action trung gian cộng lại. Với protocol submit đã có sẵn trong prompt, baseline dành phần lớn objective cho action là lựa chọn hợp lý hơn để kiểm chứng trước.

### 5.2. Khác biệt so với công thức SmartAD

SmartAD dùng weighted-token CE, chia tổng loss có trọng số cho tổng trọng số token, với hệ số reason/action/final = 1/1,5/2. Nếu áp dụng trực tiếp, số token reasoning vẫn ảnh hưởng tổng tỷ trọng. Baseline này thay đổi normalization sang mean theo nhóm và mean theo trajectory.

Đây là **thiết kế điều chỉnh từ SmartAD**, không phải tái lập nguyên bản. Chọn trajectory bằng rule cũng không phải student-NLL selection và chưa chứng minh capability alignment với student.

Để kiểm chứng tác dụng của thiết kế, baseline đối chứng nên dùng assistant-only CE đồng đều trên cùng dữ liệu/representation và cùng ngân sách. Có thể thêm weighted-token CE gần SmartAD làm đối chứng thứ hai. Khi so sánh, ghi rõ normalization và số token được supervise; “uniform token loss” và “uniform mean theo nhóm” là hai objective khác nhau.

<!-- pagebreak -->

## 6. Batching, windows và DDP không được đổi objective

Công thức trên định nghĩa objective ở mức trajectory đầy đủ. Trainer có thể xử lý từng turn hoặc nhiều đoạn để phù hợp bộ nhớ, nhưng không được âm thầm chuyển sang trung bình theo số token hay trung bình độc lập của mỗi turn.

Nếu trajectory được chia thành nhiều sample, mỗi target token chỉ được supervise **một lần**. Token lịch sử xuất hiện lại để làm context phải mask; tránh tính loss nhiều lần vì overlap. Giữ Nₙ,ᵍ và Zₙ theo toàn trajectory đã biểu diễn, không tính lại theo riêng từng đoạn.

Có thể viết contribution của sample/đoạn s thuộc trajectory n:

$$q_{n,s}=\sum_{g\in G_n}\frac{\lambda_g}{Z_nN_{n,g}}\sum_{i\in I_{n,g,s}}\ell_{n,i},\qquad L_n=\sum_s q_{n,s}.$$

Điều kiện là các Iₙ,ᵍ,ₛ phân hoạch Iₙ,ᵍ: không trùng target và không bỏ target. Nếu có N trajectory và K sample/đoạn trong toàn bộ tập, lấy sample đồng đều và dùng sample loss (K/N) × qₙ,ₛ sẽ cho kỳ vọng đúng bằng mean loss trên N trajectory. Nếu sampling theo cách khác, phải tính lại trọng số theo xác suất lấy mẫu; không mặc định sample mean bằng trajectory mean.

Cắt bỏ token hoặc dùng context ngắn hơn sẽ thay đổi target/context được học; đó là một thay đổi dataset/training policy phải ghi rõ. Tài liệu này chưa quyết định context window, packing hay số turn được giữ. Không được tuyên bố giữ nguyên objective của full-history nếu context đã bị rút ngắn.

### 6.1. Yêu cầu triển khai loss

- Teacher forcing: dự đoán token i từ các token trước i. Shift logits và labels theo causal LM đúng một lần; shift nhóm/weights tương ứng. Tránh shift kép nếu model/trainer đã thực hiện.
- Dùng per-token CE với `reduction='none'`, `ignore_index=-100`. Không lấy scalar loss mặc định của model rồi nhân lại hệ số.
- Tính tổng CE theo nhóm và reduction bằng FP32, kể cả khi forward dùng mixed precision. Quy tắc này không tự giải quyết bất ổn số học bên trong model.
- Padding không có label, không có group weight. Attention mask và loss mask là hai khái niệm khác nhau; masked observation vẫn có thể được attention đọc làm context.
- Với DDP, giữ normalization theo sample/trajectory nhất quán giữa các rank; không lấy trung bình đơn giản của các rank có batch size khác nhau. Batch cuối và gradient accumulation phải tính đúng theo số sample đóng góp.
- Không sinh labels cho user/tool text chỉ vì template biểu diễn tool observation thành user block. Nguồn role/structured fields phải được đối chiếu với render.
- Packing nếu dùng phải giữ ranh giới trajectory, tránh cross-task attention và giữ denominator của từng trajectory. Đây là yêu cầu triển khai, chưa xác nhận backend hỗ trợ.

Mục tiêu là loss và gradient tương đương giữa chạy một GPU và chia cùng effective batch trên hai GPU, trong sai số số học cho phép. DDP không tự bảo đảm điều này nếu custom reduction hoặc weighting sai.

## 7. Validation offline và lựa chọn checkpoint

Dùng 50 task held-out đã chốt; loại toàn bộ attempts của các task đó khỏi train. Final test là SWE-bench Verified 500 task, không dùng để chọn trọng số/checkpoint.

Teacher-forced validation dùng cùng template, phân vùng, normalization và hệ số với training. Báo cáo riêng:

- **Action CE:** mean của mean CE nhóm A trên các validation trajectory; tín hiệu chính để chọn checkpoint cho phase 1.
- **Submit CE:** mean CE nhóm F; dùng cùng Action CE để phát hiện trade-off, không xem là patch correctness.
- **Reasoning CE và Content CE:** theo dõi sự thay đổi, không thay thế metric action.
- **Weighted total loss:** objective đầy đủ theo công thức ở mục 5.
- Số token/trajectory có mặt trong từng nhóm và số target không phân vùng được.

Quy tắc khởi đầu: chọn checkpoint có Action CE thấp nhất; nếu ngang nhau trong độ chính xác báo cáo, ưu tiên Submit CE thấp hơn, rồi checkpoint sớm hơn. Nếu Submit CE hoặc các kiểm tra protocol xấu đi rõ rệt, cần xem lại cấu hình thay vì kết luận chỉ từ total loss.

Có thể bổ sung student sinh next action trên held-out teacher states để đo parse/schema validity, Bash syntax và action-class agreement. Phép đo này vẫn offline, không cần rollout toàn task; không phải phép đo CE ở trên. Command executability chỉ được xác nhận khi có environment phù hợp; command động/dependency chưa kiểm tra phải báo chưa xác minh. Lần chuẩn bị tài liệu này không sinh action hoặc chạy command teacher/student.

<!-- pagebreak -->

## 8. Kiểm chứng bắt buộc trước training

1. Mỗi target assistant token có đúng một group; context và padding đều `labels=-100`. Không có token F bị tính thêm vào A.
2. Tool-call tokens khớp template chính thức, gồm nhiều calls, Unicode, command nhiều dòng và quotes; không đổi payload command.
3. Marker submit trong prompt, reasoning, content hay observation không nhận nhãn F. Submit thất bại ở giữa trajectory không nhận nhãn F.
4. Terminal submit thiếu observation cuối vẫn hợp lệ. Không tạo token observation hoặc reference patch để làm target.
5. Token IDs của input khớp tokenization toàn text; kiểm tra riêng offsets của special tokens và ranh giới span.
6. Toy numerical check xác nhận mean theo nhóm, nhóm rỗng, masking và causal shift. Lặp thêm token reasoning có cùng mean CE không đổi tỷ lệ hệ số reasoning; nhân tất cả λ bằng cùng một hằng số không đổi loss.
7. Các đoạn target không overlap; cộng contributions tái tạo loss full-trajectory trên context giống nhau. Kiểm chứng scaling nếu trainer lấy sample theo turn/window.
8. Một lần backward cho thấy gradient hữu hạn và adapter thực sự cập nhật. So sánh reduction một GPU/hai GPU với cùng effective batch, rồi mới chạy dài.

Đây là checklist dành cho trainer sẽ triển khai; không phải các kiểm tra đã chạy trong lần tạo tài liệu.

## 9. Trạng thái dữ liệu và giới hạn kết luận

Baseline đã chọn 4.000 trajectory từ 4.000 task khác nhau, mỗi task ban đầu có ít nhất hai successful attempts. Dữ liệu ở `data/processed/mini_swe_agent_rule_baseline_4000/`; validation held-out có 50 task. Các record hiện có normalized messages, text render, token count và action candidates; **chưa có training labels, token group masks hoặc trainer weighted loss**.

Tổng text đã serialize của baseline khoảng 191,9 triệu token, trung vị 40.715 token/trajectory. Các số này không phải số target tokens được supervise và không phải ngân sách runtime đã đo. Giảm hệ số reasoning không giảm chi phí forward/backward của context còn nằm trong input. Chưa khẳng định 4.000 trajectory × 1 epoch vừa quota Kaggle 30 giờ.

Máy local chỉ xử lý dữ liệu, tokenizer, mã nguồn và điều khiển KRun; training dự kiến chạy trên Kaggle. Chưa có phiên GPU, training, rollout hoặc kết quả student nào được tạo bởi tài liệu này.

Success-only SFT học hành vi teacher trên teacher states. Nó chưa chứng minh student phục hồi được từ lỗi của chính mình, biết tự chọn thời điểm submit trong rollout, hoặc giải bug tốt hơn. Tác dụng của loss weighting phải được đo; các token dễ đoán trong câu submit không nên được dùng làm bằng chứng thay thế năng lực giải task.

## 10. Nguồn và provenance

1. **SmartAD, mục 3.3 — Segment-Weighted Agent Loss:** https://aclanthology.org/2026.findings-acl.1349.pdf
2. **mini-SWE-agent — release v2.4.6:** https://github.com/SWE-agent/mini-swe-agent/releases/tag/v2.4.6
3. **SWE-bench instance template, v2.4.6:** https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/config/benchmarks/swebench.yaml
4. **DefaultAgent.run dựng prompt, v2.4.6:** https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/agents/default.py
5. **Docker environment nhận submit, v2.4.6:** https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/environments/docker.py
6. **Teacher dataset:** NVIDIA Open-SWE-Traces v1.2, split minisweagent, revision `e192b26a43b8d09f156eeee781aef8e6bd5108ba`; teacher Qwen3.8-27B Thinking, resolved=1. https://huggingface.co/datasets/nvidia/Open-SWE-Traces
7. **Student tokenizer/template:** https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
8. **Audit submit local:** `data/interim/nvidia_success_submission_audit_2026-10-13.json`.
9. **Manifest baseline:** `data/processed/mini_swe_agent_rule_baseline_4000/task_selection_manifest.json`; SHA256 `dba8102cbd90be706a48dc244b444718998f974228077fac0a5a7ee564e628fe`.

Tài liệu được lưu ở dạng Markdown để đọc và chỉnh sửa trực tiếp. Dữ liệu trajectory không được nhúng vào tài liệu và không được commit cùng tài liệu.
