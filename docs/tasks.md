# Bảng task (nguồn sự thật về "ai đang làm gì")

Mọi agent/thành viên **đọc file này trước khi làm**, nhận task theo quy trình ở `AGENTS.md` §0.
Chỉ sửa dòng task của mình (và thêm dòng mới ở cuối). Ngày theo định dạng `DD/MM HH:MM`.

**Trạng thái:** `todo` (trống, ai cũng nhận được) · `doing` (đã có người nhận, **không đụng vào**) ·
`blocked` (ghi lý do) · `done` (ghi 1 dòng kết quả + file liên quan).

**Người nhận:** `Sơn`, `Hoàng`, có thể kèm agent, ví dụ `Sơn/Claude`.

**Lưu ý tài nguyên:**
- GPU của mỗi máy chỉ chạy **một** job nặng tại một thời điểm.
- Kết quả từng câu (`results/**.jsonl`, `summary.csv`) đã được commit, nên **không chạy lại** thứ đã
  có, chỉ pull về và dùng.

## Đang mở

| ID | Task | Người nhận | Trạng thái | Phụ thuộc | Cập nhật / kết quả |
|---|---|---|---|---|---|
| T03 | Track A – pha sinh câu trả lời (vLLM Qwen2.5-7B-AWQ) cho toàn bộ ma trận trong `scripts/run_all_compression.sh` | Sơn/Claude | doing | T02 | 18/10 14:50 – chờ T04 xong rồi bật vLLM |
| T04 | Chạy lại dry-run OpenProvence (ép `language="en"`) + LongLLMLingua (vá assert prefix) | Sơn/Claude | doing | – | 18/10 14:48 |
| T05 | Nghiên cứu hướng mới: nén mềm (encoder → ma trận nhớ → decoder), cơ chế "retrieval weight" của DeepSeek; đề xuất phương án khả thi trong 6 ngày | Sơn/Claude | doing | – | 18/10 14:40 – kết quả sẽ ở `docs/research_soft_compression.md` |
| T06 | Track B – chạy `scripts/run_retrieval.py` (BM25S, static, bge-small flat/HNSW/int8/binary) trên corpus gộp 50,7k đoạn | | todo | – | GPU trống hoặc chạy CPU (`EFFRAG_DEVICE=cpu`); ghi `results/retrieval/summary.jsonl` |
| T07 | Đo độ trễ bộ nén CPU vs GPU: `uv run python scripts/bench_device.py --n 20` hai lần, với `EFFRAG_DEVICE=cuda` và `EFFRAG_DEVICE=cpu` | | todo | – | Trả lời "không có GPU có chạy được không"; ghi `results/latency/cpu.jsonl`, `cuda.jsonl` |
| T08 | Bảng + hình: `uv run python scripts/make_figures.py` → `docs/report/tables/`, `docs/report/figures/`, `docs/slide/figures/` | Sơn/Claude | todo | T03, T06, T07 | |
| T09 | Báo cáo LaTeX: viết abstract, results, conclusion, contribution, appendix (`docs/report/sections/`) | Sơn/Claude | todo | T08 | Đã có: titlepage, introduction, related, method, implementation |
| T10 | Slide 15–30 trang trên template HUST (`docs/slide/template.pptx`, python-pptx) | | todo | T08 | Màu navy `#003366`, đỏ `#C02034`; hình lấy từ `docs/slide/figures/` |
| T11 | Kiểm tra demo Gradio `app/demo.py` (tab so sánh bộ nén + RAG end-to-end), chụp màn hình cho báo cáo/slide | | todo | – | Chạy được cả khi không có LLM (chỉ hiện phần nén) |
| T12 | Phương pháp mới (nén mềm) – cài đặt + thí nghiệm | | blocked | T05 | Chờ nhóm chốt phương án sau T05 |
| T13 | Track C – end-to-end: retriever tốt nhất (T06) top-20 → bộ nén tốt nhất (T03) → LLM trên corpus gộp | | todo | T03, T06 | P1 |
| T14 | Phân tích lỗi: khi nào bộ nén xoá mất đáp án / LLM vẫn sai dù còn bằng chứng (lấy ví dụ từ `results/compression/*.jsonl`) | | todo | T03 | Đưa vào mục Results |
| T15 | Cập nhật `README.md`, `docs/plan.md` theo kết quả cuối | | todo | T09 | |

## Đã xong

| ID | Task | Người làm | Kết quả |
|---|---|---|---|
| T01 | Setup uv, repo, dữ liệu LITM NQ-20, loader, tài liệu | Sơn/Claude | Commit `da9b092` |
| T02 | Dry-run nén cho 15 cấu hình (500 câu, vị trí 1/10/20) | Sơn/Claude | Provence τ=0.1: 4,6×, recall 0,916; Rerank top-5: 4,1×, 0,794; LLMLingua-2 r=0,25: 3,5×, 0,68 – `results/compression_dryrun/` |
