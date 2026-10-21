# Reproduce và verify trong trajectory Qwen3.8-27B

Ngày kiểm tra: 2026-10-20. Nguồn NVIDIA Open-SWE-Traces v1.2, revision
`e192b26a43b8d09f156eeee781aef8e6bd5108ba`, mini-SWE-agent.

**Kết luận:** có bằng chứng trực tiếp teacher tự tạo script tái hiện, chạy ra
lỗi, sửa implementation, chạy lại script và kiểm tra regression/edge cases.
Đây là hành vi hiện diện trong dữ liệu SFT hiện tại, chưa phải kết quả đánh giá
student sau fine-tuning.

## Phạm vi

Đối chiếu ID với `data/processed/first_run_epoch_sft/index.json` và thứ tự của
`data/processed/first_run_diverse_4000/diversity-plan.json`. Trong 16 record đầu
thuộc cohort gặp theo thứ tự nguồn, tìm được 8 ứng viên có ít nhất 3 command
nhắc tên script/từ khóa reproduction; đọc sâu 4 trường hợp dưới đây. Đây là
chọn mẫu để tìm bằng chứng, **không dùng để ước lượng tỷ lệ trên 4.000 mẫu**.

Cả 4 record đều có teacher metadata Qwen3.8-27B, thinking=true, resolved=1,
partition=train_pool. Chỉ đọc messages và tool observations đã lưu; không chạy
lại command của teacher, không đọc reference patch. Các kết quả test dưới đây
là output lịch sử trong dataset.

## Ví dụ thực tế

| Task / vị trí trong cohort (từ 1) | Tái hiện | Xác minh sau sửa |
| --- | --- | --- |
| `splitio/python-client#153` / 623 | Tạo `/tmp/repro.py` với thread, Event và task chậm. Output: `task fully done: False`, thời gian chờ `0.000 s`, script báo exit code 1. | Sau sửa thời điểm `task_done()`, chạy lại: `task fully done: True`, chờ `1.000 s`, `repro exit: 0`. Bộ `tests/tasks/`: **19 passed**. |
| `yt-project/unyt#73` / 2331 | Tạo `/tmp/repro.py`, dùng assertion so shape với NumPy. Chạy đúng PYTHONPATH cho ra `AssertionError: BUG: axis=-1 ignored!`, returncode=1. | Truyền `axis=axis` vào `np.stack`, chạy lại cùng repro: `All checks passed!`, returncode=0. Thêm `/tmp/edge.py`; output cuối `All edge cases passed!`. |
| `vedro-universe/vedro#104` / 2572 | Tạo file scenario rỗng trong script `/tmp/repro/repro.py`, gọi loader. Output: `FAILED with OSError: could not get source code`. | Chạy lại với `PYTHONPATH=/testbed`: `SUCCESS: loaded empty module`. Script `full_check.py` kiểm file rỗng/newline/comment và assertion rewriting: `ALL PASS`. Chạy `tests/`: **922 passed, 45 warnings**. |
| `audreyr/cookiecutter#1920` / 516 | Tạo `/tmp/repro.py`, mock lần nhập đầu trả None, lần sau trả giá trị. Trước sửa: `Result: None`, prompt chỉ gọi 1 lần. | Khi import đúng bản đã sửa: `Result: 'required_value'`, prompt gọi 2 lần. Test mục tiêu **1 passed**; nhóm test rộng vẫn có **6 failed, 47 passed**, teacher kiểm tra lại trên bản chưa sửa và cũng thấy 6 lỗi đó. |

Hai ví dụ ở vị trí 516 và 623 thuộc prefix đã train trước checkpoint 695
(1.390 trajectory), không chỉ nằm trong phần dự định train tiếp. Hai ví dụ ở
2331 và 2572 nằm trong cohort nhưng chưa thuộc prefix 2.104 mẫu tại step 881.

## Chất lượng và giới hạn

- Có xử lý lỗi môi trường: nhiều script trong `/tmp` import nhầm checkout
  `/workspace/...`; teacher kiểm tra `module.__file__`, dùng PYTHONPATH hoặc
  đồng bộ bản code trước khi chạy lại.
- Script kiểm thử cũng có lỗi: ở unyt, bản edge-case đầu dùng hai mảng khác
  shape, rồi có expected value sai; teacher sửa script qua nhiều lượt.
- Exit code 0 không tự chứng minh hết lỗi. Ví dụ Vedro catch exception rồi
  in `FAILED` nhưng process vẫn trả 0; Splitio có command `echo` phía sau làm
  returncode của shell là 0 dù script báo exit code 1. Kết luận dựa cả output.
- Không phải mọi test đều pass: Splitio có test integration lỗi, có dấu hiệu
  thiếu Redis; Cookiecutter có nhóm lỗi tồn tại cả khi stash bản sửa. Không
  diễn giải `resolved=1` thành toàn bộ test trong trajectory đều pass.
- Issue đã mô tả lỗi/steps; bằng chứng cho thấy teacher chuyển chúng thành
  script thực thi và kiểm tra kết quả, không đo khả năng tự khám phá bug mới.
- Cần rollout/evaluation student để biết model 4B đã học được hành vi này tới
  đâu. Lần kiểm tra này không đo tỷ lệ reproduce/verify trên toàn corpus.

## Bằng chứng

`evidence.json` giữ ID, task, vị trí cohort, source coordinates, SHA256 của part
nguồn và trích messages nguyên bản với chỉ số từ 0. Tool observations được giữ
nguyên theo thứ tự; assistant có nhiều tool calls nên không gán mọi observation
cho từng call riêng lẻ. Không trích reference patch.
