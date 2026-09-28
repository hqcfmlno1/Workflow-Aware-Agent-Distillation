# Teacher Trajectory Collection

Tài liệu này mô tả cách collector dùng OmniCode và SWE-agent để sinh, lưu và tổng hợp một teacher trajectory.

Mỗi attempt có runner timeout mặc định `600 giây` cho SWE-Agent và cleanup timeout độc lập `60 giây`. Tổng wall-clock tối đa vì vậy xấp xỉ `660 giây`. SWE-ReX được cấu hình startup timeout `300 giây`, SWE-Agent có agent tool timeout `600 giây`, mỗi action mặc định `60 giây`, và mỗi teacher API request mặc định `120 giây`. Collector còn dừng task khi chi phí vượt `$0.15` hoặc số teacher API call vượt `30`. Các budget này độc lập với nhau.

## 1. Luồng sinh một trajectory

```text
Chọn task
    |
    v
Tạo kế hoạch chạy theo workflow x language x task_id
    |
    v
Reset output của attempt và ghi nhận các container Docker đang tồn tại
    |
    v
Khởi động SWE-agent với task prompt, repository, mode và teacher model
    |
    v
SWE-ReX khởi động runtime/container cho repository
    |
    v
SWE-agent gọi teacher API và thực hiện các action trong repository
    |
    v
Lưu observation, action, lịch sử hội thoại và patch
    |
    v
Kết thúc task hoặc phát hiện timeout/infra error/interruption
    |
    v
Dọn container được tạo trong attempt
    |
    v
Đọc trajectory, patch, model statistics và verifier result
    |
    v
Tạo record.json rồi append vào records.jsonl và manifest.jsonl
```

### Bước 1: Chọn task

Collector đọc các dataset OmniCode theo `workflow` và `language`, sau đó chọn số lượng task được yêu cầu cho từng nhóm. `seed` giúp việc sampling có thể reproduce. `selection_manifest.jsonl` lưu các task đã được chọn; `manifest.jsonl` và trạng thái cũ được dùng để tránh chạy lại các run đã hoàn tất.

Có thể truyền `task_id` cụ thể hoặc loại trừ task bằng `--skip-task-ids`.

### Bước 2: Tạo attempt độc lập

Mỗi task/attempt có một thư mục riêng:

```text
teacher_dataset/runs/<task_id>/<workflow>/attempt_001/
```

Collector xóa output tạm của attempt đó trước khi chạy lại. Việc retry là ở cấp task; attempt bị gián đoạn hoặc lỗi hạ tầng sẽ chạy lại từ đầu task, không nối tiếp trajectory dở dang.

### Bước 3: Chạy SWE-agent và gọi teacher API

Collector gọi `baselines/sweagent/sweagent_regular.py` với:

- task dataset và `task_id`;
- mode tương ứng với workflow và language;
- teacher model;
- API base và API key từ environment;
- thư mục output của attempt.

Trước khi chạy batch, có thể prebuild image để cài sẵn SWE-ReX:

```powershell
uv run python scripts/prebuild_swerex_images.py `
  omnicodeorg/omnicode:elastic_logstash_base
```

Script tạo image hậu tố `-swerex`, cài `swe-rex==1.3.0`, chạy preflight `swerex-remote --version`, rồi mặc định retag image đó bằng tên OmniCode gốc để SWE-agent tự sử dụng image đã prebuild. Dùng `--no-retag` nếu chỉ muốn giữ tag hậu tố. Prebuild image chỉ cần thực hiện một lần cho mỗi image OmniCode; mỗi task vẫn tạo container riêng để tránh state leak. Khi dùng `--group-by-image --workers N`, các task dùng cùng image có thể chạy song song trong tối đa `N` container. Collector chỉ dọn container sau khi toàn bộ task trong nhóm image đã kết thúc.

Ví dụ cấu hình batch dài, có thể dừng chủ động:

```powershell
uv run python scripts/collect_teacher_trajectories.py `
  --languages python java cpp `
  --workflows bugfixing test_generation style_review review_response `
  --target-per-group 100000 `
  --workers 2 `
  --group-by-image `
  --prebuild-images `
  --delete-image-after-group `
  --prune-build-cache-after-group
```

Nhấn `Ctrl+C` một lần để ngừng nhận task mới và chờ task đang chạy kết thúc. Nhấn lần hai để yêu cầu dừng ngay các runner đang hoạt động.

SWE-agent dùng SWE-ReX để tạo runtime cho repository. Trong lúc agent hoạt động, mỗi vòng lặp gồm model response, tool/action, environment observation và quyết định tiếp theo của agent.

### Bước 4: Lưu artifact

Sau khi SWE-agent kết thúc, collector tìm các artifact chính:

- `.traj`: trajectory JSON do SWE-agent sinh;
- `.patch`: patch cuối nếu runner tạo được file patch;
- `all_preds.jsonl`: fallback để lấy `model_patch`;
- `stdout.log` và `stderr.log`: log chạy harness;
- `verifier.json`: kết quả benchmark nếu verifier đã được chạy.

Nếu runtime không khởi động, API không được gọi và trajectory có thể là `null`. Đây là lỗi hạ tầng, không phải model failure.

### Bước 5: Tạo và commit record

Collector tổng hợp artifact thành `record.json`. Chỉ sau khi record được ghi thành công, collector mới append record vào:

```text
teacher_dataset/records.jsonl
teacher_dataset/manifest.jsonl
```

Vì vậy hai file JSONL này là append-only và có thể dùng để resume sau crash hoặc dừng batch.

## 2. Cấu trúc lưu trữ

```text
teacher_dataset/
  selection_manifest.jsonl
  manifest.jsonl
  records.jsonl
  runs/
    <task_id>/
      <workflow>/
        attempt_001/
          metadata.json
          record.json
          stdout.log
          stderr.log
          verifier.json              # tùy chọn
          sweagent_output/
            all_preds.jsonl
            <task_id>/
              <task_id>.traj
              <task_id>.patch        # tùy chọn
```

`record.json` là bản ghi thuận tiện cho training/evaluation; `.traj` và log raw được giữ lại để debug và phân tích sâu hơn.

## 3. Schema của một record

Mỗi dòng trong `records.jsonl` là một JSON object gồm các trường sau:

| Trường | Kiểu thường dùng | Giải thích ngắn |
|---|---|---|
| `task_id` | `string` | ID duy nhất của task trong dataset OmniCode. |
| `language` | `string` | Ngôn ngữ của repository/task, hiện gồm `python`, `java` hoặc `cpp`. |
| `workflow` | `string` | Loại workflow được chạy, ví dụ `bugfixing`, `test_generation`, `style_review` hoặc `review_response`. |
| `status` | `string` | Kết quả thực thi của collector/harness, chẳng hạn `completed_success`, `completed_failure`, `budget_exhausted`, `infra_failed`, `interrupted` hoặc `timeout`. |
| `success` | `boolean` hoặc `null` | Kết quả thành công; ưu tiên lấy từ verifier, nếu chưa có verifier thì dùng tín hiệu submit/harness và có thể là `null` khi lỗi hạ tầng. |
| `verifier_score` | `number` hoặc `null` | Điểm raw do OmniCode verifier trả về; để `null` nếu verifier chưa chạy hoặc không có điểm số. |
| `tokens_input` | `integer` hoặc `null` | Số token teacher nhận trong attempt, lấy từ model statistics của SWE-agent. |
| `tokens_output` | `integer` hoặc `null` | Số token teacher sinh ra trong attempt. |
| `api_calls` | `integer` hoặc `null` | Số lần SWE-agent gọi teacher API. |
| `cost` | `number` hoặc `null` | Chi phí được model adapter ghi nhận cho instance/attempt. |
| `runtime` | `number` | Thời gian wall-clock của attempt, tính bằng giây từ lúc collector bắt đầu chạy task đến lúc tổng hợp record. |
| `trajectory` | `object` hoặc `null` | Nội dung `.traj`, thường gồm `history`, `info`, `replay_config` và `environment`. |
| `final_patch` | `string` hoặc `null` | Patch cuối của agent, lấy từ `.patch` hoặc fallback từ `model_patch` trong `all_preds.jsonl`. |

## 4. Cách đọc trạng thái và success

`status` và `success` trả lời hai câu hỏi khác nhau:

- `status`: collector có hoàn tất được attempt hay gặp lỗi trong quá trình chạy?
- `success`: agent có giải quyết task theo tiêu chí thành công hay không?

Một run có thể có `status = completed_success` nhưng `success = false` nếu agent hoàn tất workflow nhưng patch không đạt yêu cầu. `budget_exhausted` nghĩa là model đã dùng hết cost/API-call budget; đây không phải lỗi Docker và collector sẽ không tự chạy lại cùng run. Ngược lại, `infra_failed`, `timeout` và `interrupted` thường có `success = null` vì chưa có đủ evidence để đánh giá model.

Khi `verifier.json` tồn tại, `success` và `verifier_score` nên được xem là nguồn đánh giá chính. Nếu verifier chưa chạy, `completed_success` chỉ có nghĩa là SWE-agent/harness đã kết thúc, không khẳng định benchmark đã pass.

## 5. Điều kiện để một trajectory dùng được

Một trajectory đầy đủ tối thiểu nên có:

1. `task_id`, `language`, `workflow` và `status`.
2. `trajectory` không rỗng và có history/action/observation.
3. `tokens_input`, `tokens_output`, `api_calls` và `cost` nếu teacher API trả statistics.
4. `final_patch` nếu workflow tạo ra patch.
5. `success` hoặc `verifier_score` sau khi chạy verifier.
6. Log và metadata đủ để truy lại attempt khi có lỗi.

Các record lỗi hạ tầng vẫn được lưu để phân tích và retry, nhưng không nên đưa vào tập imitation chính nếu không có trajectory hợp lệ.

## 6. Timeout và cleanup

Collector áp dụng các timeout độc lập cho một task như sau:

```text
0–300s    SWE-ReX startup limit
0–600s    SWE-Agent tool-execution limit
0–60s     mỗi action/tool call
0–120s    mỗi teacher API request
0–600s    runner process hard limit
0–60s     terminate process tree, cleanup container và ghi record
```

Ngoài timeout, mỗi task mặc định có tối đa `$0.15` chi phí model và `30` teacher API call. Nếu runner vượt hard limit, collector terminate toàn bộ process tree. Nếu một action, API request, cost limit, call limit hoặc agent tool budget chạm giới hạn riêng, SWE-Agent xử lý lỗi và thoát trước runner timeout.

Ở chế độ thường, collector dọn container sau từng attempt. Ở chế độ group-by-image, cleanup được hoãn tới khi toàn bộ task dùng image đó kết thúc để không xóa nhầm container của worker khác. Image chỉ bị xóa khi bật `--delete-image-after-group`; build cache chỉ được prune một lần sau toàn bộ batch.
