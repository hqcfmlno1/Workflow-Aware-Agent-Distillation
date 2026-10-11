# NVIDIA Open-SWE-Traces × OmniCode: kiểm tra trùng task

Ngày kiểm tra: 2026-10-10. Có task trùng; cần loại khỏi dữ liệu student trước khi đánh giá trên OmniCode.

## Nguồn và phạm vi

- NVIDIA: https://huggingface.co/datasets/nvidia/Open-SWE-Traces/tree/e192b26a43b8d09f156eeee781aef8e6bd5108ba
- OmniCode: https://huggingface.co/datasets/seal-research/OmniCode/tree/87de6c4e0ecdb132bfcba96ee2e0a624ce6f26ad
- Đã đọc đủ 216/216 shard Parquet của NVIDIA: 521.712 trajectory, 42.248 khóa repository/PR duy nhất. Danh sách shard đối chiếu với API `siblings` của revision đã chốt.
- OmniCode: 494 base instance (Python 273, Java 109, C++ 112) và 415 style instance (Python 144, Java 124, C++ 147).
- Phạm vi training hiện định dùng: `data/sweagent/qwen35_122b/swe-rebench-v2/`, 9/9 shard, 20.334 trajectory, 10.511 khóa repository/PR, trong đó 7.348 trajectory `resolved == 1`. Các số này chưa lọc category bug-fix.

## Phương pháp

Đọc các cột `instance_id`, `repo`, `trajectory_id`, `language`, `resolved` từ Parquet qua HTTP range requests; không cần tải toàn bộ hội thoại. Ghép theo repository viết thường và số PR, chuẩn hóa các ID dạng `repo__name-123`, `repo__name_123` và `repo_name_pr123`. Với style task, ưu tiên `original_instance_id` hoặc `pull_number` để tránh nhầm chỉ số biến thể thành số PR.

Cùng repository không được tính là trùng task. Phạm vi student dự định dùng có 3 repository chung với OmniCode: `keras-team/keras`, `scrapy/scrapy`, `simdjson/simdjson`, nhưng chỉ có một PR trùng.

Với ba PR tìm được, đọc thêm patch tham chiếu của một trajectory mỗi PR để kiểm chứng. Bỏ dòng `index ...` và phần mô tả sau dấu `@@ ... @@` khi so patch, giữ nguyên đường dẫn file và nội dung thay đổi. Riêng task thuộc phạm vi training chính đã đọc metadata của cả ba trajectory, xác nhận category `bug-fix`, teacher `Qwen3.5-122B-A10B`, `enable_thinking = false` và patch tham chiếu.

## Kết quả

| OmniCode base instance | NVIDIA instance | Trajectory NVIDIA toàn bộ | Trong Qwen3.5-122B / SWE-agent |
| --- | --- | ---: | ---: |
| `scrapy__scrapy-6050` | `scrapy__scrapy-6050` | 21 | 3 |
| `scrapy__scrapy-6469` | `scrapy__scrapy-6469` | 13 | 0 |
| `simdjson__simdjson_2016` | `simdjson__simdjson-2016` | 14 | 0 |

Tổng cộng 3/494 base instance, liên quan 48 trajectory NVIDIA; không tìm thấy khóa repository/PR trùng với 415 style instance. Base instances được OmniCode tái sử dụng cho bug fixing, test generation và review response, nên cần giữ việc loại trừ ở cấp task gốc khi mở rộng workflow.

### Đúng cấu hình hiện tại

`scrapy__scrapy-6050` sửa lỗi backward compatibility của `RetryMiddleware.EXCEPTIONS_TO_RETRY` khi truy cập qua instance. Cả ba trajectory đều `resolved == 1` và `metadata.category == bug-fix`. Lọc successful trajectory hoặc bug-fix không tự loại được contamination này.

| Trajectory ID | Shard |
| --- | --- |
| `65cc59b6-5497-44fe-bdec-e6844c0e5856` | `train-00002-of-00009.parquet` |
| `5460dd3c-5f6f-4579-949b-c6f9ad267790` | `train-00005-of-00009.parquet` |
| `334ad630-6eef-4af2-8882-4c23d422342a` | `train-00007-of-00009.parquet` |

Loại ba trajectory này giảm tập successful từ 7.348 xuống 7.345, trước các bộ lọc khác.

### Kiểm chứng patch

- Scrapy PR 6050: patch tham chiếu khớp sau chuẩn hóa metadata git diff.
- Scrapy PR 6469: patch mẫu khớp sau chuẩn hóa; NVIDIA gắn category `feature-request` cho mẫu kiểm tra.
- simdjson PR 2016: cùng repository/PR; bảy file trong patch OmniCode có nội dung patch khớp sau chuẩn hóa với NVIDIA. NVIDIA có thêm thay đổi ở `value_iterator-inl.h`, nên toàn bộ patch không giống hệt. Mẫu NVIDIA gắn category `feature-request`; OmniCode problem statement là yêu cầu hỗ trợ comma-separated documents trong `iterate_many`.
- `feature-request` nói trên là category của mẫu đã kiểm tra, không phải kết quả quét category của mọi trajectory thuộc PR đó.

## Cách áp dụng

Giữ task evaluation OmniCode và loại các trajectory trùng khỏi train, trước khi chia train/validation. Với đúng Qwen3.5-122B/SWE-agent hiện tại, cần loại toàn bộ `scrapy__scrapy-6050` (ba trajectory). Nếu bổ sung teacher/harness khác, loại cả ba khóa repository/PR tìm được. Có thể đối chiếu dữ liệu mới với toàn bộ 494 benchmark keys để tránh chỉ dựa vào blacklist của snapshot hiện tại.

Artifacts trong `reports/data_overlap_2026-10-10/`:

- `summary.json`: phạm vi, revisions, số đếm.
- `overlapping_trajectories.csv`: 48 trajectory trùng và vị trí nguồn.
- `training_exclusions.json`: các khóa cần loại và ba trajectory ID của cấu hình chính.
- `omnicode_base_keys.csv`: 494 khóa repository/PR của benchmark.
- `patch_checks.json`: kết quả kiểm chứng patch mẫu.
- `source_index_sha256.json`: SHA-256 của sáu file OmniCode và 216 file index cột đã đọc từ NVIDIA; đây là hash index JSON, không phải hash nguyên Parquet. Dữ liệu tải và index đầy đủ được giữ cục bộ tại `/tmp/swe-overlap-audit/`.

## Giới hạn

Đây là kiểm tra exact task theo repository/PR, có kiểm chứng patch cho các task trùng. Chưa quét semantic similarity trên toàn bộ problem statement/reference patch để phát hiện PR khác nhau nhưng giải cùng issue, repository đổi tên hoặc fork. Không có kết luận về dữ liệu pretraining của student. Số đếm áp dụng cho hai revision đã ghi, không bảo đảm các revision mới giữ nguyên.
