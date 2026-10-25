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
| LongLLMLingua 730t (token, theo câu hỏi) | 670 | 4,3× | 62,4 / 63,2 / 67,6 | 64,4 |
| Rerank top-5 (chọn tài liệu) | 701 | 4,1× | 64,2 / 62,0 / 62,0 | 62,7 |
| Dense top-5 / BM25 top-5 | 702 / 705 | 4,1× | 57,4 / 56,0 / 56,4 · 47,6 / 45,4 / 45,8 | 56,6 / 46,3 |
| LLMLingua-2 r=0,25 (xoá token) | 827 | 3,5× | 36,6 / 40,8 / 37,6 | 38,3 |
| Closed-book / Oracle | 0 / 113 | – / 25,5× | không phụ thuộc vị trí | 38,8 / 92,6 |

- Provence và LongLLMLingua là hai bộ nén không giảm accuracy có ý nghĩa thống kê ở cả 3 vị trí; hai
  bộ ngang nhau (so cặp n.s.), nhưng Provence nén nhanh hơn 3× (200 so với 625 ms/câu trên GPU).
- Reader hiện đại gần như không còn *lost in the middle* (chênh 5,8 điểm so với 21,6 của
  GPT-3.5) → lợi ích "nén làm tăng accuracy ở giữa" của LongLLMLingua không tái lập được.
- Không cần GPU: nén trên CPU (rerank 0,2 s, Provence 4,9 s/câu) + reader qua API.
- Truy hồi: bge-small int8 hit@5 76,2 (chỉ mục 18,6 MB) so với BM25 55,2.
- End-to-end (bge-small int8 top-20 trên kho 50,7k → nén → 7B): full 66,0; rerank-5 63,0; Provence
  62,4 (4,6×, không giảm có ý nghĩa); LLMLingua-2 36,4.
- Nén mềm (PISCO, reader Mistral-7B nf4): 160 vector thay 2.875 token (18×), TB 48,2 so với 60,3
  của văn bản trên cùng reader; giảm 9 / 10 / 17 điểm ở vị trí 1 / 10 / 20 (đều có ý nghĩa). Rerank top-5
  trước PISCO: 58,5 với 40 vector (72×), so với 62,9 của văn bản rerank top-5.
- Nén mềm encode-once của nhóm (Qwen2.5-1.5B, 2 LoRA, 8 vector/đoạn, 15k mẫu, 1 epoch): **không qua
  cổng go/no-go** (dev: 22 so với closed-book 19). Chẩn đoán: một đoạn vàng dạng 8 vector chỉ cho 23%
  so với 72% dạng văn bản. Biến thể hybrid 48,5% (5,5×) ≈ chỉ đọc top-1 dạng văn bản (49,2%).
- Reader nhỏ (Qwen2.5-1.5B) nhạy vị trí (53,2 / 42,0 / 40,8) → ở đây Provence **tăng** accuracy
  7,5 điểm (có ý nghĩa ở cả 3 vị trí), đúng hiệu ứng LongLLMLingua báo cáo với GPT-3.5.

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

### Demo Gradio

Cần làm xong bước 1–2 ở trên (môi trường + dữ liệu). Gradio nằm trong nhóm `demo`, đã được
`uv sync` cài mặc định.

```bash
# (Tuỳ chọn) bật LLM để demo sinh câu trả lời – ví dụ vLLM cục bộ, cùng model với thí nghiệm:
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --served-model-name qwen2.5-7b-instruct \
  --max-model-len 4096 --port 8000
# rồi trong .env: LLM_BASE_URL=http://localhost:8000/v1  LLM_MODEL=qwen2.5-7b-instruct

# Chạy demo → mở http://localhost:7860
uv run python app/demo.py

# Không có GPU / GPU đang bận: ép chạy bộ nén và retriever trên CPU
EFFRAG_DEVICE=cpu uv run python app/demo.py

# vLLM và demo chung một GPU 12 GB: đẩy LLMLingua-2/LongLLMLingua sang CPU (vLLM dùng
# --gpu-memory-utilization 0.55 --enforce-eager), các bộ nén còn lại vẫn chạy GPU
EFFRAG_DEMO_LLMLINGUA_DEVICE=cpu uv run python app/demo.py

# Mở cho máy khác trong mạng truy cập, đổi cổng
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7861 uv run python app/demo.py
```

- **Tab "So sánh bộ nén":** chọn câu hỏi (200 câu NQ đầu), vị trí đoạn vàng và các bộ nén
  (Full, Rerank top-5, LLMLingua-2, LongLLMLingua, Provence, OpenProvence). Mỗi bộ nén hiện số
  token trước/sau, thời gian nén, đáp án còn trong ngữ cảnh hay không, và câu trả lời của LLM; câu
  được giữ tô xanh, câu bị loại gạch xám.
- **Tab "RAG end-to-end":** nhập câu hỏi tiếng Anh → truy hồi top-k bằng BM25S hoặc bge-small trên
  kho gộp 50,7k đoạn → nén → LLM trả lời.
- Không có LLM (chưa điền `.env` hoặc server tắt) demo vẫn chạy, chỉ bỏ phần câu trả lời.
- Lần đầu chọn mỗi bộ nén sẽ tải model từ Hugging Face (đặt `HF_HOME` sang ổ còn trống). Provence
  và OpenProvence dùng `trust_remote_code`: nếu terminal hỏi `Do you wish to run the custom code?`
  thì trả lời `y`. Trên CPU, LongLLMLingua và LLMLingua-2 mất khoảng 13–15 s/câu, Provence ~5 s.

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
app/                 demo Gradio (app/demo.py, xem mục "Demo Gradio")
results/             kết quả (commit)      data/  dữ liệu + cache (không commit)
docs/                research.md, plan.md, requirements.md, report/, slide/
```
