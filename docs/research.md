# Nghiên cứu hướng làm (khảo sát 18/10/2026)

Tất cả arXiv ID đã kiểm tra qua `export.arxiv.org`, HF repo qua `huggingface.co/api`. Số liệu đã
đọc từ bảng gốc của bài báo (ghi rõ bảng). Thứ chưa kiểm chứng ghi `UNVERIFIED`.

## 1. Vì sao chọn hướng này

| Phương án | Ưu | Nhược | Kết luận |
|---|---|---|---|
| **Lập trình: benchmark nén ngữ cảnh + truy hồi hiệu quả trên NQ-20 (LITM)** | Có số liệu công bố trên *đúng* bộ dữ liệu & giao thức (3 bài) → đáp ứng yêu cầu "so sánh với nghiên cứu gần đây cùng bộ dữ liệu"; model ≤ 600M chạy được trên máy nhóm; có demo trực quan | Generator gốc (GPT-3.5-0613) đã ngừng → chỉ so độ chênh/hình dạng | **Chọn** |
| Lập trình, góc tiếng Việt (UIT-ViQuAD2.0, Belebele `vie_Latn`) | Mới (audit đa ngôn ngữ 2608.26175 chưa có tiếng Việt) | Ít số liệu để so, tốn công xử lý dữ liệu | Làm **mục phụ** "Vietnamese spot-check" nếu dư thời gian |
| Tiểu luận 8 bài | Rủi ro thấp | Trần điểm thấp hơn, không demo | Không chọn |
| Soft compression / KV-cache (xRAG, PISCO, OSCAR, REFRAG, kvpress) | Rất "hot" | Cần generator 7B chạy local, can thiệp nội bộ model → không hợp API + GPU 12GB bận | Chỉ đưa vào Related work |

**Đóng góp dự kiến:** chưa tìm thấy báo cáo nào đánh giá thế hệ pruner 2025–26 (Provence,
XProvence, OpenProvence, Zilliz semantic-highlight) trên giao thức theo-vị-trí của LITM NQ-20 →
nhóm bổ sung dòng đó, phân tích pruner có "làm phẳng" đường cong chữ U không, và đánh đổi
accuracy ↔ token ↔ latency ↔ $. (Diễn đạt "chúng tôi không tìm thấy", không khẳng định "chưa ai làm".)

## 2. Phương pháp sẽ dùng

### 2a. Nén ngữ cảnh (đều trả về text → dùng được với LLM qua API)

| Method | arXiv / venue | Checkpoint | Kích thước | Ghi chú |
|---|---|---|---|---|
| Truncate, top-k (BM25/dense/reranker) | – | – | – | baseline |
| LLMLingua-2 | 2403.12968, Findings ACL'24 | `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` | 559M | ✅ đã chạy được |
| LongLLMLingua | 2310.06839, ACL'24 | `llmlingua` + `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | bài gốc dùng Llama-2-7B → ghi rõ khác biệt |
| Provence | 2501.16214, ICLR'25 | `naver/provence-reranker-debertav3-v1` | 435M | ✅ đã chạy được; CC-BY-NC-ND; remote code |
| XProvence (đa ngôn ngữ) | 2601.18886, ECIR'26 | `naver/xprovence-reranker-bgem3-v2` | 568M | cần spacy `xx_sent_ud_sm` |
| OpenProvence | – (MIT, `hotchpotch/open_provence`) | `hotchpotch/open-provence-reranker-v1-gte-modernbert-base`, `…-xsmall-v1` (37M, chạy CPU) | 37–150M | |
| Zilliz semantic-highlight | – (blog HF 01/2026) | `zilliz/semantic-highlight-bilingual-v1` | 567M | cùng base bge-reranker-v2-m3 với XProvence → chọn 1 nếu thiếu thời gian |
| RECOMP (extractive) | 2310.04408, ICLR'24 | `fangyuan/nq_extractive_compressor` | ~110M | tuỳ chọn |
| Sentinel | 2505.23277 | `ReRaWo/Sentinel` (.pkl!) + Qwen2.5-0.5B | 0.5B | tuỳ chọn, file pickle không an toàn |
| EXIT | 2412.12559, Findings ACL'25 | `doubleyyh/exit-gemma-2b` (LoRA, gemma gated) | 2.5B | tuỳ chọn, cắt đầu tiên |

Loại: xRAG, CompAct-7B, PISCO, OSCAR, CLaRa (cần 7B làm generator); CORE, EnComp (không có checkpoint).

### 2b. Truy hồi hiệu quả

| Thành phần | Checkpoint / thư viện | Kích thước |
|---|---|---|
| BM25S (2407.03618) | `bm25s` | – |
| Static embedding | `sentence-transformers/static-retrieval-mrl-en-v1`, `minishlab/potion-retrieval-32M` | rất nhỏ |
| Dense nhỏ | `BAAI/bge-small-en-v1.5`, `ibm-granite/granite-embedding-small-english-r2` (47M) | |
| Matryoshka (MRL) | `Qwen/Qwen3-Embedding-0.6B` (2506.05176), dim 1024→64 | 596M |
| Lượng tử hoá int8 / binary + rescoring | `sentence_transformers.quantization` + `faiss-cpu` | |
| Late interaction | `pylate` + `mixedbread-ai/mxbai-edge-colbert-v0-17m`, `lightonai/GTE-ModernColBERT-v1` | 17M / 149M |
| Learned sparse | `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill` | 67M |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2`, `Qwen/Qwen3-Reranker-0.6B` | 23M / 0.6B |

Ba thiết lập: (a) chọn doc trong 20 doc/câu; (b) corpus gộp ~50.7k đoạn (đo index size, ms/query,
top-k hit); (c) NanoNQ NDCG@10 để so với model card (GTE-ModernColBERT 0.7645,
mxbai-edge-colbert 0.7086, static-retrieval-mrl 0.4534 – *số tự công bố của tác giả*).

Đo thử ban đầu (BM25S, corpus gộp 50.714 đoạn, 200 câu): build 2.2 s, 0.39 ms/query,
top-1/5/20 answer-hit = 0.19 / 0.51 / 0.90.

## 3. Số liệu công bố để đối chiếu (cùng dữ liệu LITM NQ-20, generator GPT-3.5-Turbo)

Cột = vị trí đoạn vàng. Accuracy (%).

| Nguồn (bảng) | Method | 1st | 5th | 10th | 15th | 20th | Tokens |
|---|---|---|---|---|---|---|---|
| Lost in the Middle 2307.03172 (bảng 20 docs) | gpt-3.5-turbo-16k | 75.7 | 57.3 | 54.1 | 55.4 | 63.1 | – |
| LITM (bảng closed-book/oracle) | gpt-3.5-turbo: closed-book **56.1**, oracle **88.3** | | | | | | |
| LongLLMLingua 2310.06839 (`main_results_qa`) | Original prompt | 75.7 | 57.3 | 54.1 | 55.4 | 63.1 | 2,946 |
| ↳ 2x | LongLLMLingua | 77.2 | 72.9 | 70.8 | 70.5 | 70.6 | 1,429 |
| ↳ 2x | SBERT top-k | 72.5 | 67.9 | 63.3 | 65.0 | 66.2 | 1,549 |
| ↳ 2x | BM25 top-k | 53.7 | 49.3 | 47.9 | 49.9 | 46.9 | 1,545 |
| ↳ 2x | LLMLingua | 39.7 | 39.5 | 40.4 | 37.1 | 42.3 | 1,410 |
| ↳ 2x | Selective-Context | 45.4 | 39.0 | 33.8 | 33.5 | 41.5 | 1,478 |
| ↳ 4x | LongLLMLingua | 75.0 | 71.8 | 71.2 | 71.2 | 74.7 | 748 |
| ↳ 4x | SBERT top-k | 66.9 | 61.1 | 59.0 | 61.2 | 60.3 | 808 |
| ↳ 4x | OpenAI embedding top-k | 63.8 | 64.6 | 65.4 | 64.1 | 63.7 | 804 |
| ↳ 4x | BM25 top-k | 40.6 | 38.6 | 38.2 | 37.4 | 36.6 | 798 |
| LLMLingua-2 2403.12968 (`nq_results`), 4x | LLMLingua-2 (không dùng câu hỏi) | 48.6 | 44.5 | 43.6 | 40.9 | 39.9 | 748 |
| ↳ 4x | LLMLingua-2 + coarse filter của LongLLMLingua | 74.0 | 70.4 | 67.0 | 66.9 | 65.3 | 739 |

Số liệu **khác thiết lập** (chỉ để thảo luận, ghi rõ "different setting"):

- FlashRAG 2405.13576 (Llama-3-8B-Instruct, e5-base-v2, top-5): NQ EM – no retrieval 22.6,
  standard RAG 35.1, LongLLMLingua 32.2, RECOMP-abs 33.1, Selective-Context 30.5.
- Provence 2501.16214 (BERGEN, Llama-2-7B, LLM-judge, top-5): NQ full 71.8 vs Provence 72.4 khi
  bỏ 62.2% context; LLMLingua-2 59.5 khi bỏ 74.0%.
- EXIT 2412.12559 (Llama-3.1-8B): NQ EM original 34.6 → EXIT 35.9.
- CORE 2508.19282 (ICML'26, Qwen2.5-14B): NQ EM full top-5 38.03 → CORE 41.02 với ~6% token.

## 4. Lưu ý giao thức (rút ra từ các bài)

- Subset cố định 500 câu × 5 vị trí, seed 42; full 2.655 câu chỉ cho cấu hình tốt nhất.
- Tự chạy lại original / closed-book / oracle với generator của nhóm; so *độ chênh* và *hình
  dạng đường cong*, không so số tuyệt đối với GPT-3.5.
- Giữ thứ tự doc gốc sau nén (bảng công bố làm vậy; "Reorder" là cột riêng).
- Pruner kiểu Provence xử lý từng đoạn độc lập → kết quả nén giống nhau giữa 5 file vị trí (chỉ
  khác thứ tự) → có thể nén 1 lần, tiết kiệm 5x (kiểm tra diff vài câu trước khi tận dụng).
- OpenProvence/Zilliz train trên dữ liệu "natural-questions-context-relevance": khả năng trùng
  test NQ-Open là `UNVERIFIED` → nêu như một caveat.
- Ghim phiên bản sentence splitter (nltk punkt, spacy) vì ảnh hưởng kết quả.

## 5. Tài liệu tham khảo chính (Related work)

| # | Bài | arXiv | Venue |
|---|---|---|---|
| 1 | Lost in the Middle: How Language Models Use Long Contexts | 2307.03172 | TACL 2024 |
| 2 | Compressing Context to Enhance Inference Efficiency of LLMs (Selective Context) | 2310.06201 | EMNLP 2023 |
| 3 | LLMLingua: Compressing Prompts for Accelerated Inference of LLMs | 2310.05736 | EMNLP 2023 |
| 4 | LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression | 2310.06839 | ACL 2024 |
| 5 | LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression | 2403.12968 | Findings ACL 2024 |
| 6 | RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation | 2310.04408 | ICLR 2024 |
| 7 | xRAG: Extreme Context Compression for RAG with One Token | 2405.13792 | NeurIPS 2024 |
| 8 | FlashRAG: A Modular Toolkit for Efficient RAG Research | 2405.13576 | WWW 2025 |
| 9 | EXIT: Context-Aware Extractive Compression for Enhancing RAG | 2412.12559 | Findings ACL 2025 |
| 10 | Provence: efficient and robust context pruning for RAG | 2501.16214 | ICLR 2025 |
| 11 | OSCAR: Online Soft Compression And Reranking | 2504.07109 | ICLR 2026 |
| 12 | Sentinel: Decoding Context Utilization via Attention Probing for Efficient LLM Context Compression | 2505.23277 | preprint |
| 13 | Less Is More: Elevating RAG via Performance-Driven Context Compression (CORE) | 2508.19282 | ICML 2026 |
| 14 | REFRAG: Rethinking RAG based Decoding | 2509.01092 | 2025 |
| 15 | XProvence: Zero-Cost Multilingual Context Pruning for RAG | 2601.18886 | ECIR 2026 |
| 16 | Lost in Compression: A Controlled Cross-Lingual Audit of Extractive Prompt Compressors | 2608.26175 | 2026 |
| 17 | The Attribution-Compression Frontier in RAG | 2609.14245 | GroundLM @ EMNLP 2026 |
| 18 | Prompt Compression for Large Language Models: A Survey | 2410.12388 | NAACL 2025 |
| 19 | BM25S: Orders of magnitude faster lexical search via eager sparse scoring | 2407.03618 | 2024 |
| 20 | Matryoshka Representation Learning | 2205.13147 | NeurIPS 2022 |
| 21 | Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models | 2506.05176 | 2025 |
| 22 | EmbeddingGemma: Powerful and Lightweight Text Representations | 2509.20354 | 2025 |
| 23 | PyLate: Flexible Training and Retrieval for Late Interaction Models | 2508.03555 | 2025 |
| 24 | The Faiss library | 2401.08281 | 2024 |

Đọc thêm (đã kiểm tra ID): PISCO 2501.16075, BERGEN 2407.01102, CompAct 2407.09014, AttnComp
2509.17486, CLaRa 2511.18659, SeleCom 2602.15856, KVzip 2505.23416, Expected Attention/kvpress
2510.00636, vCache 2502.03771.

## 6. Rủi ro

- **GPU bận** → tất cả model ≤ 600M, nén offline 1 lần rồi lưu; demo có chế độ CPU
  (OpenProvence-xsmall, static embedding, BM25S).
- **Chi phí API** → subset 500, cache, temperature 0; full context ≈ 500×5×2.9k ≈ 7.3M token/cấu hình.
- **Remote code / pickle** → ghim revision, không load `.pkl` lạ.
- **License** → Provence/XProvence CC-BY-NC-ND (dùng học thuật OK); EmbeddingGemma/Gemma cần xin quyền.
- **Nhóm 2 người** (đề yêu cầu 3–4) → cần xác nhận với giảng viên.
