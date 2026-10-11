# Response format: Qwen3.8-27B / NVIDIA v1.2

Kiểm tra trực tiếp ngày 2026-10-11 trên revision `e192b26a43b8d09f156eeee781aef8e6bd5108ba` của `nvidia/Open-SWE-Traces`, config v1.2, split minisweagent.

## Phạm vi và kết quả

Đọc toàn bộ `messages`, `tools` và `metadata.teacher_model` của hai shard:

- `data/minisweagent/qwen38_27b/scale-swe/train-00000-of-00022.parquet`: 2.500 trajectory, 120.202 assistant message.
- `data/minisweagent/qwen38_27b/swe-rebench-v2/train-00000-of-00022.parquet`: 2.500 trajectory, 135.758 assistant message.

Tổng 5.000 trajectory, 255.960 assistant message. Đây là hai shard đầu tiên, không phải mẫu ngẫu nhiên và không phải kiểm tra nội dung cả 44 shard.

Mỗi message có schema `role`, `content`, `reasoning_content`, `tool_calls`. Trong cả hai shard:

- Tất cả assistant message có `reasoning_content` không rỗng.
- Tất cả assistant message có `tool_calls` không rỗng; tất cả function name được quan sát là `bash`.
- `content` là phần lời nói hiển thị, có thể rỗng: 223.837/255.960 assistant message có content không rỗng.
- Metadata teacher đều `name = Qwen3.8-27B`, `enable_thinking = true`, `reasoning_effort = xhigh`.
- Có nhiều tool calls trong một assistant message, tối đa 5 trong hai shard; phải giữ đầy đủ danh sách thay vì lấy call đầu.
- Không có `<think>` trong content của assistant ở hai shard; reasoning được lưu riêng.
- Chỉ một assistant content có bash fence. Không dùng regex tìm code block trong content làm bộ trích action mặc định cho dữ liệu này.

Ví dụ thật của trajectory `b82d4956-5621-438a-957e-f5d43f94245f` có reasoning, content và hai bash calls trong cùng response. Assistant message cuối có command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`. Đây là ví dụ termination qua bash, không phải một tool native tên submit.

## Ý nghĩa của “nguyên response”

Dataset lưu nội dung phản hồi assistant theo các trường đã chuẩn hóa, bao gồm reasoning và tool calls, chứ không chỉ final patch hoặc command. Có thể dựng lại representation cho SFT từ các trường này.

Không có raw provider response để đối chiếu lossless/byte-for-byte. Message schema không giữ usage, finish_reason, logprobs, và tool response không có trường tool_call_id. Không thể kết luận không có truncation hoặc preprocessing ở upstream. Khi cần nối tool result với nhiều tool calls, phải kiểm chứng quy tắc thứ tự/cách collector chuẩn hóa trước khi chuyển sang format khác.

## Hệ quả cho action-only phase 1

Dùng `tool_calls[].function.name` và `tool_calls[].function.arguments` làm nguồn action; serialize qua student template phù hợp rồi mask các span action và cấu trúc/termination cần thiết. Giữ reasoning/content/observations làm context nếu đó là thiết kế được chọn. Không mặc định action của NVIDIA mini-SWE-agent nằm trong bash block, cũng không suy ra runner mini-SWE-agent nào cũng dùng cùng protocol chỉ từ tên harness.

Nhận định trước đó trong hội thoại về mini-SWE-agent thường dùng bash block chưa mô tả đúng dataset đang dùng. Cần căn cứ vào native tool_calls có sẵn của dữ liệu này và kiểm tra target harness cụ thể.

## Bằng chứng

- `reports/response_inspection_2026-10-11/summary.json`: số đếm theo từng source và các giới hạn.
- `reports/response_inspection_2026-10-11/example_assistant_response.json`: response đầu và response kết thúc thực tế, không rút gọn các trường trong artifact.
- Dữ liệu hai shard đã đọc được giữ cục bộ trong `/tmp/swe-overlap-audit/response_samples_*.json`, không đưa vào Git.
