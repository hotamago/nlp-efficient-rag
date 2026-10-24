# Kế hoạch & phân công (18/10 → CN 25/10/2026)

- **Sơn (A):** Track A – nén ngữ cảnh, pipeline đánh giá, LLM; báo cáo phần Method / Experiments / Results.
- **Hoàng (B):** Track B – truy hồi hiệu quả, demo Gradio; báo cáo phần Introduction / Related work /
  Retrieval results; slides.
- **Cả hai:** phân tích lỗi, thống nhất số liệu, đọc chéo báo cáo, tập thuyết trình (ai cũng phải trình bày).

Quy tắc: tick `[x]` + 1 dòng kết quả chính khi xong (xem `AGENTS.md` §8).

## 0. Đã xong (18/10)

- [x] Môi trường uv (Python 3.11, torch 2.14, transformers 5.20, llmlingua 0.2.2), repo GitHub
- [x] Tải LITM NQ-20 (2.655 câu × 5 vị trí), loader + corpus gộp (50.714 đoạn)
- [x] Compressor: full, closedbook, oracle, truncate, dense_topk, llmlingua2, provence (dry-run OK)
  - Provence th=0.1 (n=20, gold@0): ratio 4.9x, ctx_recall 0.95
- [x] BM25S trên corpus gộp (n=200): top-1/5/20 hit 0.19/0.51/0.90, 0.39 ms/query
- [x] LLM client OpenAI-compatible + cache, schema kết quả, tests

## 0b. Trạng thái cuối (19/10) và thay đổi so với kế hoạch

Toàn bộ thí nghiệm đã xong. Số liệu nằm trong `results/`, báo cáo ở `docs/report/main.pdf` (28 trang),
slide ở `docs/slide/slides.pdf` (18 trang), gói Overleaf tạo bằng `scripts/overleaf_zip.py`.

- [x] Track A (reader Qwen2.5-7B-Instruct-AWQ, 500 câu × vị trí 1/10/20): Provence τ=0,1 đạt 64,0 so với
  65,3 của full (4,6×, không có ý nghĩa); LongLLMLingua 64,4 (4,3×, ngang Provence nhưng chậm hơn 3×); rerank top-5 62,7; LLMLingua-2 r=0,25 38,3 ≈ closed-book 38,8.
  Chỉ đo 3 vị trí {0, 9, 19} thay vì 5 để tiết kiệm GPU.
- [x] Track B: bge-small int8 hit@5 76,2 (chỉ mục 18,6 MB), BM25S 55,2. Qwen3-Embedding và ColBERT
  không chạy (tốn GPU, bge-small đã đủ để rút ra kết luận).
- [x] Track C: top-20 bge-small int8 → nén → reader 7B: full 66,0; rerank-5 63,0; Provence 62,4;
  LLMLingua-2 36,4.
- [x] CPU vs GPU: rerank 196/16 ms, Provence 4,9/0,2 s mỗi câu.
- [x] Mở rộng ngoài kế hoạch ban đầu (đã thống nhất, T12/T16–T18): nén mềm. PISCO 48,2 so với 60,3
  văn bản (Mistral-7B). Bộ nén encode-once tự huấn luyện (Qwen2.5-1.5B) không qua cổng dừng; chẩn
  đoán: 8 vector/đoạn không giữ được đáp án. Phụ: reader 1.5B nhạy vị trí, Provence tăng 7,5 điểm.
- Bỏ: A6 (RECOMP, Sentinel, EXIT, tiếng Việt), B3 NanoBEIR.

## 1. Ma trận thí nghiệm

### Track A – Nén ngữ cảnh (LITM NQ-20, n=500, seed 42, vị trí {0,4,9,14,19})

| Nhóm | Cấu hình | Ưu tiên |
|---|---|---|
| A0 tham chiếu | closedbook, oracle, full | P0 |
| A1 baseline | truncate (budget 1450 / 730), dense_topk k=10/5 (bge-small), BM25 top-k, reranker top-k | P0 |
| A2 LLMLingua-2 | rate 0.5 / 0.25; + "coarse" dense top-k trước rồi LLMLingua-2 | P0 |
| A3 LongLLMLingua | Qwen2.5-0.5B-Instruct, 2x / 4x | P1 |
| A4 pruner mới | Provence, XProvence-v2, OpenProvence (base, xsmall), Zilliz; threshold {0.05, 0.1, 0.3} | P0 (Provence, OpenProvence), P1 (còn lại) |
| A5 hai tầng | retriever/reranker top-k → pruner | P1 |
| A6 tuỳ chọn | RECOMP-extractive, Sentinel, EXIT; kiểm tra attribution (trích dẫn `[doc i]`, 100 câu); spot-check tiếng Việt (XProvence + UIT-ViQuAD2.0) | P2 |

Kết quả cần có: bảng theo vị trí (đặt cạnh bảng LongLLMLingua/LLMLingua-2 trong
`docs/research.md` §3), đồ thị accuracy vs token, accuracy vs latency, đường cong theo vị trí,
phân tích lỗi (khi nào pruner xoá mất đáp án).

### Track B – Truy hồi hiệu quả

| Thiết lập | Cấu hình | Chỉ số |
|---|---|---|
| B1 chọn doc trong 20 doc | BM25S, static-retrieval-mrl, bge-small, Qwen3-Embedding (MRL 1024/256/64), MiniLM cross-encoder, Qwen3-Reranker | gold hit@k theo vị trí |
| B2 corpus gộp 50.7k | BM25S, static, bge-small/granite-small, Qwen3-Emb MRL, float/int8/binary+rescore, PyLate mxbai-edge-colbert, sparse opensearch-v3 | answer-hit@1/5/20, MRR@10, index MB, build s, ms/query (p50/p95) |
| B3 NanoNQ | các model trên qua `NanoBEIREvaluator` | NDCG@10, so với model card |

### Track C – End-to-end & demo

- Retriever tốt nhất (B2) top-20 → compressor tốt nhất (A) → LLM, trên corpus gộp.
- Demo Gradio `app/demo.py`: câu hỏi → các đoạn truy hồi → chọn compressor (so sánh song song,
  highlight câu giữ lại) → câu trả lời, token tiết kiệm, latency, $. Có chế độ CPU.

## 2. Lịch theo ngày

| Ngày | Sơn (A) | Hoàng (B) |
|---|---|---|
| CN 18/10 | ✅ setup, nghiên cứu, tài liệu | Đọc README/AGENTS/research, `uv sync`, chạy test; chọn & thống nhất `LLM_MODEL` |
| T2 19/10 | A0 + A1 (dry-run rồi chạy LLM), bảng theo vị trí đầu tiên | B1; tải sẵn model (static, Qwen3-Emb, rerankers, colbert) |
| T3 20/10 | A2 + A3 | B2 (corpus gộp, quantization, MRL, ColBERT, sparse) + B3 |
| T4 21/10 | A4 (Provence family, quét threshold) | Demo Gradio v1 |
| T5 22/10 | A5; chạy full 2.655 câu cho 2–3 cấu hình tốt nhất; A6 nếu kịp | Vẽ đồ thị (A+B), demo v2 (tab retrieval) |
| T6 23/10 | Viết Method, Implementation (thư viện, tham số), Results vs bảng công bố | Viết Introduction, Related work, Retrieval results; nháp slides |
| T7 24/10 | Phân tích lỗi, Kết luận, hoàn thiện báo cáo | Slides 15–30, quay video demo dự phòng, mục đóng góp thành viên |
| CN 25/10 | Đọc soát, tập thuyết trình 15', **nộp** | (cùng) |

## 3. Dàn ý báo cáo (15–30 trang, dạng bài báo)

1. Giới thiệu bài toán (chi phí token, lost-in-the-middle, câu hỏi nghiên cứu, đóng góp)
2. Nghiên cứu liên quan (token/prompt compression; context pruning; soft & KV compression; efficient retrieval)
3. Phương pháp (pipeline, các compressor/retriever, giao thức đánh giá)
4. Cài đặt (thư viện, checkpoint + revision, dữ liệu, tham số, phần cứng, LLM + ngày chạy)
5. Kết quả & phân tích (so với LITM / LongLLMLingua / LLMLingua-2; trade-off; vị trí; lỗi)
6. Kết luận & hướng phát triển
7. Đóng góp của từng thành viên
8. Tài liệu tham khảo
