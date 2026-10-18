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
| T08 | Bảng + hình: `uv run python scripts/make_figures.py` → `docs/report/tables/`, `docs/report/figures/`, `docs/slide/figures/` | Sơn/Claude | doing | T03, T07 | |
| T09 | Báo cáo LaTeX: viết abstract, results, conclusion, contribution, appendix (`docs/report/sections/`) | Sơn/Claude | doing | T08 | 18/10 17:30 – xong results (trừ nén mềm), CPU, conclusion (trừ RQ5), contribution, appendix; còn abstract + mục nén mềm |
| T10 | Slide Beamer 15–30 trang theo style `docs/slide/template.tex` (HUST đỏ/trắng) → `docs/slide/slides.tex` | Sơn/Claude | doing | T08 | Hình lấy từ `docs/slide/figures/`. File template gốc có tên/MSSV nhóm khác → gitignore, không commit |
| T12 | P0 – đánh giá nén mềm có sẵn: `naver/pisco-mistral` (4-bit, sdpa) trên LITM 500×{1,10,20} + full/closed-book/oracle của Mistral cùng prompt PISCO + lai "rerank top-5 → PISCO" | Sơn/Claude | doing | T03 (GPU) | 18/10 19:50 – PISCO TB 48,2 (47,8/52,8/44,0, 18×) vs văn bản Mistral 60,3 (56,6/63,2/61,2), giảm có ý nghĩa ở cả 3 vị trí; đang chạy closed-book/oracle/rerank-5 |
| T16 | P1 – tự huấn luyện nén mềm "encode-once" (ý tưởng của nhóm): Qwen2.5-1.5B-Instruct làm encoder+decoder, LoRA, 8 vector nhớ / 128 token, dữ liệu Tevatron NQ-train (~15k, vị trí vàng ngẫu nhiên, k∈{5,10,20}); biến thể lai top-1 dạng text (ý tưởng indexer/CSA của DeepSeek) | Sơn/Claude | doing | – | 18/10 17:30 – smoke 10 bước: 7 s/bước, 5 GB VRAM (1 epoch ≈ 3,7 h); đã sửa lỗi gradient encoder khi bật checkpointing (`37d0aa0`); chạy full sau T12 |
| T17 | Cổng dừng P1 (21/10): 200 câu Tevatron-dev; đi tiếp nếu acc ≥ closed-book + 10 điểm và ≥ 70% full-text của reader 1.5B, nếu không thì dừng P1 và viết phân tích thất bại | Sơn/Claude | todo | T16 | |
| T18 | Đánh giá P1 trên LITM 500×{1,10,20}: soft16x, lai, + các dòng của reader 1.5B (full, closed-book, oracle, 2–3 bộ nén cứng tốt nhất dùng lại context đã nén) | | todo | T17 | |
| T13 | Track C – end-to-end: retriever tốt nhất (T06: bge-small int8) top-20 → bộ nén tốt nhất (T03) → LLM trên corpus gộp | | todo | T03 | P1 |
| T15 | Cập nhật `README.md`, `docs/plan.md` theo kết quả cuối | Sơn/Claude | doing | T09 | 18/10 – README có mục Kết quả chính (Track A/B, CPU); chờ số nén mềm |

## Đã xong

| ID | Task | Người làm | Kết quả |
|---|---|---|---|
| T01 | Setup uv, repo, dữ liệu LITM NQ-20, loader, tài liệu | Sơn/Claude | Commit `da9b092` |
| T02 | Dry-run nén cho 15 cấu hình (500 câu, vị trí 1/10/20) | Sơn/Claude | Provence τ=0.1: 4,6×, recall 0,916; Rerank top-5: 4,1×, 0,794; LLMLingua-2 r=0,25: 3,5×, 0,68 – `results/compression_dryrun/` |
| T06 | Track B – `scripts/run_retrieval.py` trên corpus gộp 50,7k đoạn, 500 câu | Sơn/Claude | answer-hit@5: BM25 55,2 · static-256 55,8 (49,5 MB) · bge-small 76,2 (74,3 MB) · int8 76,2 (18,6 MB) · binary+rescore 73,4 (2,3 MB) – `results/retrieval/summary.jsonl` |
| T04 | Chạy lại dry-run OpenProvence (`language="en"`) + LongLLMLingua (vá assert prefix) | Sơn/Claude | OpenProvence τ=0,1: 2,6×, recall 0,964; LongLLMLingua 730t (n=200, vị trí 10): 4,3×, 0,775 |
| T05 | Nghiên cứu nén mềm + cơ chế "retrieval weight" của DeepSeek | Sơn/Claude | DeepSeek V3.2/V4/V4.1-Flash: *lightning indexer* (chọn top-k token, weight riêng); nhóm chốt P0 + P1 có cổng dừng – `docs/research_soft_compression.md` |
| T19 | Chuyển báo cáo sang style HUST đỏ/trắng (XeLaTeX), thêm mục "Nén mềm" + DeepSeek vào related/method | Sơn/Claude | Commit `564612f`; font TeX Gyre theo tên file (chạy được trên Overleaf/tectonic) |
| T11 | Kiểm tra demo Gradio (2 tab) trên CPU + ảnh chụp | Sơn/Claude | Chạy được hoàn toàn trên CPU; sửa câu hỏi mặc định tab end-to-end; ảnh `docs/report/figures/demo*.png` (Playwright headless), đã đưa vào báo cáo §Demo và slide |
| T03 | Track A – pha sinh câu trả lời (vLLM Qwen2.5-7B-AWQ), 15 cấu hình | Sơn/Claude | TB 3 vị trí: full 65,3 · Provence τ=0,1 64,0 (4,6×, không giảm có ý nghĩa) · rerank-5 62,7 (4,1×) · LLMLingua-2 r=0,25 38,3 (≈ closed-book 38,8) · LongLLMLingua 64,0 @10 (n=200) – `results/compression/`, `main_table.csv`, `significance.csv`, `errors.csv` |
| T07 | Độ trễ bộ nén CPU vs GPU (n=20, trung vị ms/câu) | Sơn/Claude | GPU/CPU: rerank 16/196 · Provence 200/4.906 · LLMLingua-2 455/13.104 · LongLLMLingua 625/14.837; CPU chậm 12–29×. `results/latency/`, `tables/latency.tex`, báo cáo §res-cpu, slide 14 |
| T14 | Phân tích lỗi so với ngữ cảnh đầy đủ | Sơn/Claude | Provence: mất BC 1,7% · hỏng dù còn BC 8,4% · được sửa 8,8%; LLMLingua-2 r=0,25 hỏng dù còn BC 19,0% (văn bản vụn) – `scripts/error_analysis.py`, `tables/errors.tex`, báo cáo §res-errors |
