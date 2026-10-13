# Research context từ tài liệu của người dùng

Đọc ngày 2026-10-11 qua Google Drive/Google Docs connector. Đây là ghi chú đọc tài liệu để dùng cho lần thiết kế kế hoạch sau, không phải implementation plan đã được duyệt.

Nguồn: [Workflow-Specific Distillation for Interactive Coding Agents](https://docs.google.com/document/d/1sTo5FkhMAAzfOYU3qtbB05sugGP6syreCi-SnTkGnGQ/edit).

Tab literature: `t.es0d1dtnrx47`. Đã đọc cả nội dung trả về của các tab proposal, kế hoạch, observation/evaluation OmniCode, tiến độ tuần, vấn đề hiện tại và timeline. Nội dung paper bên dưới là cách người dùng tổng hợp trong tài liệu; chưa đối chiếu độc lập bản gốc, venue, công thức hoặc kết quả thực nghiệm.

## Cấu hình hiện tại, ưu tiên hơn ghi chú lịch sử

Theo hội thoại mới nhất:

- Student: Qwen3.5-4B.
- Teacher: Qwen3.8-27B Thinking.
- Harness: mini-SWE-agent.
- Main data: NVIDIA Open-SWE-Traces v1.2, dùng trajectory public.
- Training: dự kiến QLoRA trên Kaggle qua KRun.
- Workflow khởi đầu: bugfix; OmniCode vẫn là hướng evaluation đang xem xét, cần adapter mini-SWE-agent và quản lý chi phí môi trường.

Các ghi chú về GPT-5.6 Luna, student 9B, SWE-agent và tự thu teacher trajectories phản ánh giai đoạn trước. Không tự đổi cấu hình repo theo các ghi chú này. Không cần chạy teacher cho baseline SFT từ dữ liệu public; các bước teacher correction/rollout mới sẽ cần tài nguyên riêng nếu được chọn sau này.

## Mục tiêu nghiên cứu

Trong một harness cố định, tìm hiểu workflow khác nhau có behavioral characteristics nhất quán hay không, đặc trưng nào liên hệ với success, và có thể dùng chúng để distill tập trung/data-efficient hơn successful-trajectory imitation đồng đều hay không.

Ba câu hỏi chính:

1. Có workflow-conditioned behavioral structure đủ rõ và nhất quán không?
2. Signal nào liên quan đến success, thay vì chỉ là artifact của harness, dataset, độ khó hoặc trajectory length?
3. Signal đó có cải thiện distillation so với baseline imitation hay không?

Harness-specific execution knowledge là một phần năng lực agent, không mặc định là nhiễu cần loại. Không giả định sẵn taxonomy hành vi hoặc khẳng định pattern là đặc tính phổ quát độc lập với harness. Cross-workflow interference không còn là problem chính mặc định.

Bugfix-only là phạm vi pilot; chưa đủ để chứng minh khác biệt giữa nhiều workflow.

## Các hướng literature đã đọc

| Work được ghi trong tài liệu | Ý tưởng được người dùng tổng hợp | Điểm cần giữ cho thảo luận sau |
| --- | --- | --- |
| SmartAD | Chọn successful trajectory bằng student NLL; segment-weighted loss | Capacity mismatch; so sánh uniform, reasoning-emphasized và action-emphasized supervision bằng thực nghiệm |
| When Agents Look the Same | Response Pattern Similarity và Action Graph Similarity | Đo behavior bên cạnh success; phân biệt mandatory behavior với thói quen teacher |
| Student-Centered Distillation / SCoRe | Cold start, sửa earliest student error, short-horizon RL | Supervision theo student state distribution; cần student rollout và environment support |
| SOD | Step-wise on-policy distillation, kiểm soát drift/noisy teacher signal | Ghi chú nguồn: white-box teacher, không dùng trực tiếp trong setting hiện tại |
| Structured Distillation of Web Agent Capabilities Enables Generalization | Tổ chức synthetic tasks/trajectories | Nguồn ý tưởng về data construction, chưa mặc định là loss mới |
| P-BRIDGE | Adaptive Trajectory Remediation và Strategy Insight Extraction | Phân biệt sửa continuation với recovery sau action sai đã thực thi; chi phí teacher repair/rejection sampling |
| Sub-goal Distillation | Segment trajectory thành sub-goal; high-level và low-level agent | Procedural/planning representation; chưa chốt dùng hai student |
| Magnet | Graph translation cho multi-turn tool-use data synthesis | Ghi chú hiện có chủ yếu motivation; cần đọc bản gốc trước khi dùng method |
| Structured Agent Distillation | Reasoning/action loss được normalize và weight riêng | Tránh reasoning dài lấn át action; cần xác minh công thức gốc khi thiết kế loss |
| DGPO | SFT cold start rồi policy optimization với teacher guidance | Không mặc định RL phù hợp ngay với student nhỏ hoặc ngân sách hiện tại |
| SKILL-KD | Contrast student failure/teacher success để tạo prompt-level skill | Không update weights; cần phân biệt với mục tiêu QLoRA fine-tuning |
| Coding Agents are Strong Prompt Optimizers / CASD | Offline corpus analysis để tạo behavioral rules/textual skill | Nguồn ý tưởng khám phá pattern; prompt optimization không đồng nhất với weight distillation |

## Những phân biệt quan trọng

- Student-aware: chọn supervision phù hợp capacity hoặc state distribution của student.
- Structure-aware: tách chức năng reasoning, action, termination, sub-goal để tổ chức supervision.
- Failure-aware: dùng failure thực tế của student để xác định và sửa điểm yếu.
- Error avoidance: tránh đi vào bad state.
- Trajectory correction: thay bad continuation bằng continuation đúng.
- Post-error recovery: agent đã thực hiện action sai, phải nhận ra vấn đề từ evidence, revise/backtrack và trở lại tiến trình giải đúng.

Ví dụ recovery được đề xuất trong ghi chú: `TEST_FAIL → INSPECT_FAILURE → REVISE_HYPOTHESIS → EDIT → RETEST`. Đây là pattern minh họa cần kiểm chứng, không phải taxonomy bắt buộc. Test failure cũng có thể là bước chẩn đoán có chủ đích; không tự gắn mọi test fail là agent error.

Post-error recovery là candidate gap của nhóm literature đã khảo sát, chưa phải novelty claim đã xác minh trên toàn bộ literature.

## Nguyên tắc cho lần thiết kế kế hoạch tiếp theo

- Đo base student trong harness trước; nhu cầu Action-focused SFT và quy mô dữ liệu phải dựa vào evidence.
- Có baseline successful-trajectory SFT trước khi quy kết cải thiện cho behavior-aware method.
- Giữ observations làm context; xác định rõ reasoning/action/termination spans được supervise và cách serialize cho Qwen3.5.
- Thinking bật trong metadata chưa chứng minh reasoning luôn hiện diện hoặc được serialize đúng trong mọi mẫu.
- Giữ failed trajectory phục vụ phân tích dù baseline chỉ train successful samples; teacher failure không đồng nghĩa student failure.
- Chưa chốt loss weighting, selection, recovery supervision, preference learning hoặc RL.
- So sánh công bằng teacher pool, split, context, supervised-token/compute budget và inference settings khi đánh giá method.
- Full-agent verifier outcome là metric chính; validation loss là tín hiệu theo dõi training.
- Phân tích steps, search/read/edit/test, verification, invalid action, termination, vòng edit-test và recovery; kiểm soát task difficulty và length.
- Mở rộng workflow sau bugfix nếu muốn trả lời câu hỏi cross-workflow.
- Kiểm tra termination/protocol của mini-SWE-agent với OmniCode; ghi chú cũ về submit hai lần không được tự áp sang harness mới.

## Câu hỏi đang để mở

Action-focused SFT cải thiện hành vi agent đến mức nào? Bao nhiêu task/token là đủ? Ưu tiên task coverage hay nhiều attempts cùng task? Reasoning và action nên normalize/weight thế nào? Có evidence recovery hữu ích trong dữ liệu public không? Làm sao phân biệt signal workflow với teacher style hoặc harness constraints? Những hướng cần student rollout/teacher correction có vừa compute budget không?

Nguồn tham chiếu hiện có trong repo: `configs/datasets/nvidia_mini_swe_agent_v1_2_action_focused_sft.json`, `docs/data_overlap_v1_2_2026-10-11.md`. Chưa sửa Google Doc hoặc triển khai phương pháp distillation mới trong lần đọc này.
