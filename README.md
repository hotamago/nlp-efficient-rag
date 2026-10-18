# Efficient RAG: Nén ngữ cảnh & Truy hồi hiệu quả cho hỏi đáp với LLM

BTL môn **IT4772 – Xử lý ngôn ngữ tự nhiên** (GV: Lê Thanh Hương).
Đề tài 13: *Efficient LLM/RAG: Token Compression, Context Compression và Efficient Retrieval*.
Dạng: **lập trình** (phần mềm + demo + báo cáo dạng bài báo).

| Thành viên | Vai trò chính |
|---|---|
| Nguyễn Hoàng Sơn | Track A – nén ngữ cảnh, đánh giá, phần Method/Results |
| Võ Việt Hoàng | Track B – truy hồi hiệu quả, demo Gradio, Related work, slides |

**Deadline nộp: Chủ nhật 25/10/2026** (báo cáo PDF 15–30 trang + 15–30 slides + demo, trình bày 15 phút).

## 1. Bài toán (nói ngắn)

RAG nhét 10–20 đoạn văn truy hồi được vào prompt → **tốn token, chậm, đắt**, và LLM còn bị
"lạc giữa ngữ cảnh" (*Lost in the Middle*: đáp án nằm giữa prompt thì độ chính xác tụt mạnh).

Câu hỏi nghiên cứu:

1. Các bộ **nén ngữ cảnh** mới (2024–2026) giảm được bao nhiêu token mà **không mất độ chính xác**?
   Có giúp LLM **bớt nhạy với vị trí** của đoạn chứa đáp án không?
2. Các kỹ thuật **truy hồi hiệu quả** (BM25S, static embedding, Matryoshka, lượng tử hoá
   int8/binary, HNSW) đánh đổi chất lượng ↔ tốc độ ↔ bộ nhớ thế nào?
3. Kết quả có khớp với số đã công bố (GPT-3.5-Turbo) trên cùng bộ dữ liệu không?
4. Không có GPU thì có làm được không?
5. Nén mềm "mã hoá một lần" (mỗi đoạn → 8 vector) có thay được văn bản không?

## 2. Hướng làm

```
câu hỏi ──► Retriever (Track B) ──► 20 đoạn ──► Compressor (Track A) ──► LLM (API) ──► đáp án
              BM25S / dense / MRL /               full / truncate / top-k /
              int8-binary / HNSW                  LLMLingua-2 / LongLLMLingua /
                                                  Provence / OpenProvence / PISCO / nén mềm của nhóm
```

- **Dữ liệu chính:** NQ-Open multi-document của *Lost in the Middle* (2.655 câu hỏi × 20 đoạn
  Wikipedia; nhóm dùng 500 câu, đoạn vàng ở vị trí 1/10/20). Đây là bộ mà **Lost in the Middle, LongLLMLingua,
  LLMLingua-2** đã công bố số liệu theo từng vị trí → so sánh trực tiếp được (yêu cầu của BTL).
- **Điểm mới của nhóm:** chưa thấy công trình nào đánh giá các bộ *pruner* thế hệ mới
  (Provence – ICLR'25, OpenProvence) trên đúng giao thức theo-vị-trí này; thêm nén mềm (PISCO và
  một bộ nén encode-once tự huấn luyện trên Qwen2.5-1.5B).
  Nhóm chạy chúng cạnh các baseline cũ và đối chiếu với bảng đã công bố.
- **Generator:** 1 LLM cố định qua API tương thích OpenAI (cấu hình trong `.env`).
- **Chỉ số:** accuracy (đáp án xuất hiện trong output), `ctx_recall` (đáp án còn trong context
  sau nén), số token, tỉ lệ nén, độ trễ nén/LLM, chi phí ước tính; Track B: top-k hit, MRR,
  NDCG@10 (NanoNQ), kích thước index, ms/query.
- **Demo:** Gradio – nhập câu hỏi, xem từng compressor giữ lại câu nào (highlight), token tiết
  kiệm, độ trễ và câu trả lời.

Chi tiết nghiên cứu: [`docs/research.md`](docs/research.md) · Kế hoạch & phân công:
[`docs/plan.md`](docs/plan.md) · Yêu cầu BTL: [`docs/requirements.md`](docs/requirements.md) ·
Quy ước cho agent/thành viên: [`AGENTS.md`](AGENTS.md).

## Kết quả chính

Reader `Qwen/Qwen2.5-7B-Instruct-AWQ` (vLLM, temperature 0), 500 câu × vị trí vàng 1/10/20,
accuracy = best-subspan-EM. Chi tiết: báo cáo `docs/report/`, số liệu `results/`.

| Cấu hình | Token | Nén | Acc @1 / @10 / @20 | TB |
|---|---|---|---|---|
| Full (20 đoạn) | 2.875 | 1,0× | 65,6 / 62,2 / 68,0 | 65,3 |
| Provence τ=0,1 (chọn câu) | 621 | 4,6× | 63,8 / 62,6 / 65,6 | 64,0 |
| Rerank top-5 (chọn tài liệu) | 701 | 4,1× | 64,2 / 62,0 / 62,0 | 62,7 |
| LLMLingua-2 r=0,25 (xoá token) | 827 | 3,5× | – | 38,3 |
| Closed-book / Oracle | – | – | 38,8 / 92,6 | – |

- Provence là bộ nén duy nhất không giảm accuracy có ý nghĩa thống kê ở cả 3 vị trí.
- Reader hiện đại gần như không còn *lost in the middle* (chênh 5,8 điểm so với 21,6 của
  GPT-3.5) → lợi ích "nén làm tăng accuracy ở giữa" của LongLLMLingua không tái lập được.
- Không cần GPU: nén trên CPU (rerank 0,2 s, Provence 4,9 s/câu) + reader qua API.
- Truy hồi: bge-small int8 hit@5 76,2 (chỉ mục 18,6 MB) so với BM25 55,2.
- Nén mềm (PISCO, reader Mistral-7B nf4): 160 vector thay 2.875 token (18×), TB 48,2 so với 60,3
  của văn bản trên cùng reader; giảm 9 / 10 / 17 điểm ở vị trí 1 / 10 / 20 (đều có ý nghĩa).

## 3. Chạy nhanh

```bash
# 1. Cài môi trường (cần uv: https://docs.astral.sh/uv/)
uv sync
cp .env.example .env              # điền LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
uv run python -c "import nltk; nltk.download('punkt_tab')"

# 2. Tải dữ liệu (~75 MB)
uv run python scripts/download_data.py

# 3. Thử nén không gọi LLM (miễn phí)
uv run python scripts/run_compression.py --method provence --dry-run --n 20 --positions 0

# 4. Chạy thật với LLM
uv run python scripts/run_compression.py --method llmlingua2 --params '{"rate": 0.25}'

# 5. Test
uv run pytest -q
```

Kết quả ghi vào `results/<track>/summary.csv` (mỗi dòng = method × tham số × vị trí vàng).

## 4. Cấu trúc thư mục

```
src/effrag/          package chính
  data.py            loader NQ-20 (Lost in the Middle) + corpus gộp (~50k đoạn)
  compression/       full, closedbook, oracle, truncate, dense_topk, llmlingua2, provence
  retrieval.py       BM25S, dense (sentence-transformers + FAISS)
  generation.py      LLM client OpenAI-compatible + cache đĩa + prompt chuẩn LITM
  metrics.py         accuracy (best_subspan_em), EM, F1
  results.py         schema kết quả chung
scripts/             download_data, run_compression, smoke_* (kiểm tra model)
tests/               pytest
app/                 demo Gradio (sẽ làm)
results/             kết quả (commit)      data/  dữ liệu + cache (không commit)
docs/                research.md, plan.md, requirements.md, report/, slide/
```
