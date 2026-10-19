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
| T12 | P0 – đánh giá PISCO (`naver/pisco-mistral`, nf4) trên LITM 500×{1,10,20} | Sơn/Claude | PISCO TB 48,2 (18×) vs văn bản cùng reader 60,3 (giảm 9/10/17 điểm, có ý nghĩa); closed-book 36,2, oracle 89,6; rerank-5→PISCO @10 56,2 (72×) vs text 63,0. `results/compression/*pisco*`, `*mistral_text*`, `tab:soft` |
| T08 | Bảng + hình (`scripts/make_figures.py`) | Sơn/Claude | Bảng chính, recall, độ trễ, truy hồi, nén mềm (+ bản tách cho slide), hình vị trí/đánh đổi/pipeline/loss nén mềm → `docs/report/{tables,figures}`, `docs/slide/{tables,figures}` |
| T10 | Slide Beamer (`docs/slide/slides.tex`) | Sơn/Claude | 17 trang, biên dịch sạch, `docs/slide/slides.pdf`; gói Overleaf: `scripts/overleaf_zip.py` |
| T16 | P1 – tự huấn luyện nén mềm encode-once (Qwen2.5-1.5B, 2 LoRA, 8 vector/128 token, 15k mẫu Tevatron NQ) | Sơn/Claude | 1 epoch, loss 7,4 → ~1,2; checkpoint resume đầy đủ mỗi 100 bước. `results/soft/run1_train.jsonl`, `scripts/train_soft.py` |
| T17 | Cổng dừng P1 (dev 200) | Sơn/Claude | KHÔNG đạt: soft 22,0 vs closed-book 19,0 / văn bản 42,0 → dừng P1. Chẩn đoán: chỉ đoạn vàng dạng 8 vector 23 vs văn bản 72; lai với đoạn vàng mở 70. `results/soft/{gate,diag}.json` |
| T18 | Đánh giá P1 trên LITM 500×{1,10,20} + baseline reader 1.5B | Sơn/Claude | Soft 25,7 ≈ closed-book 26,2; lai 48,5 (5,5×) = rerank top-1 văn bản 49,2; full 45,3; Provence 52,8 (+7,5, có ý nghĩa: reader 1.5B nhạy vị trí). `tab:soft`, `sec:res-soft` |
| T09 | Báo cáo LaTeX (`docs/report/`) | Sơn/Claude | 28 trang, biên dịch sạch (XeLaTeX/tectonic), đủ RQ1–RQ5, nén mềm, end-to-end; `docs/report/main.pdf` |
| T13 | Track C – end-to-end: bge-small int8 top-20 (kho 50,7k) → nén → Qwen2.5-7B | Sơn/Claude | Recall top-20 95,4; acc full 66,0, rerank-5 63,0, Provence 62,4 (−3,0/−3,6, không có ý nghĩa), LLMLingua-2 36,4. `results/e2e/`, `tab:e2e`, `scripts/build_retrieved.py` |
| T15 | Cập nhật `README.md`, `docs/plan.md` theo kết quả cuối | Sơn/Claude | README mục Kết quả chính (A/B/C, nén mềm, CPU); plan.md §0b trạng thái cuối |
