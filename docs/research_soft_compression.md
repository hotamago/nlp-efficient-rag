# Nghiên cứu: nén "mềm" (soft/latent) ngữ cảnh & ý tưởng encoder → ma trận nhớ → decoder

Khảo sát ngày 18/10/2026 cho đề xuất của nhóm: *"mô hình nén token cực mạnh gồm encoder (nén token
thành một ma trận thông tin dày đặc, chạy một lần) và decoder (nhận câu hỏi + ngữ cảnh + ma trận →
sinh token bình thường)"*, cùng gợi ý *"DeepSeek V4.1 Flash có cơ chế tách riêng trọng số cho truy
hồi"*.

Quy ước kiểm chứng: **[V]** = đã đọc trực tiếp trên trang arXiv/HTML của bài, model card HF, API HF
hoặc mã nguồn; **[V-tóm]** = đọc qua bản tóm tắt tự động của trang gốc (số liệu khớp bảng nhưng
nên mở PDF trước khi đưa vào báo cáo); **UNVERIFIED** = chưa kiểm chứng. Mọi arXiv ID trong file
đã kiểm tra tiêu đề qua `export.arxiv.org`.

---

## 0. Tóm tắt

1. **DeepSeek-V4.1-Flash có thật** (arXiv 2609.19969, 17/09/2026). Cái "trọng số riêng cho truy
   hồi" mà bạn nhớ là **indexer**: một mạng nhỏ có trọng số riêng, chấm điểm các mục KV
   rồi chọn **top-k** cho attention. Indexer xuất hiện từ DeepSeek-V3.2 (*lightning indexer*, DSA),
   đi tiếp sang V4 (CSA) và V4.1-Flash (CSA2 + *Hierarchical Sparse Indexer*). Đây là **chọn lọc
   cứng** (hard selection), nên về bản chất gần với reranker top-k / Provence của Track A hơn là
   nén mềm. V4.1-Flash còn có **Causal Encoder-Decoder (CED)**: decoder attend vào KV toàn cục
   chiếu từ trạng thái giữa của encoder. Đây là thứ giống ý tưởng của bạn nhất, nhưng nằm *bên
   trong một mô hình* 552B. **Engram** là bộ nhớ tra cứu n-gram tĩnh, không phải truy hồi trên ngữ
   cảnh.
2. **Ý tưởng encoder → ma trận nhớ → decoder đã có tên và có checkpoint**: ICAE, xRAG, COCOM,
   **PISCO**, **OSCAR**, CLaRa, REFRAG, 500xCompressor. Ở mức nén 16x, nếu decoder được fine-tune
   (LoRA) thì độ chính xác *ngang hoặc thấp hơn nhẹ* so với không nén (PISCO NQ 0.65 so với 0.69).
   Ở mức 128x hoặc khi giữ decoder đóng băng thì mất nhiều điểm (xRAG 39.1 so với 42.7 EM; COCOM
   frozen 0.421 so với tuned 0.519 EM). Như vậy cái lợi thật là **FLOPs/TTFT**, không phải "vừa
   nén cực mạnh vừa tăng độ chính xác".
3. **Khoảng trống phù hợp với nhóm** (chúng tôi *không tìm thấy* bài nào làm): đo nén mềm theo
   **giao thức vị trí của Lost-in-the-Middle (NQ-20)**, so trực tiếp với nén cứng (Provence,
   LLMLingua-2…) trên cùng 500 câu. Câu hỏi nghiên cứu: biểu diễn mỗi tài liệu bằng 8 vector thì
   đường cong chữ U còn không? Kèm theo: một **công thức huấn luyện ngân sách thấp** (12 GB, khoảng
   15k ví dụ, vị trí vàng ngẫu nhiên, k=10–20) cho compressor kiểu "encode-once", và biến thể
   **lai cứng→mềm**.
4. **Khuyến nghị:** (P0, không cần huấn luyện, an toàn) chạy checkpoint có sẵn `naver/pisco-mistral`
   (và `naver/oscar-qwen2-7B` nếu có quyền Llama-3.2-1B) trên LITM NQ-20 theo vị trí, kèm dòng
   full-text của chính decoder đó. (P1, đóng góp riêng, có cổng dừng) tự huấn luyện compressor
   kiểu PISCO trên **Qwen2.5-1.5B-Instruct** với prompt LITM, dữ liệu Tevatron NQ-train, một đêm GPU.
   Phương án dự phòng: nếu bước huấn luyện không qua cổng thì chỉ nộp kết quả P0 cộng phân tích.
   Để làm được phải **cắt A3 (LongLLMLingua), A6 và B3** khỏi `docs/plan.md`.

---

## 1. DeepSeek thực sự làm gì (đối chiếu với "trọng số riêng cho truy hồi")

### 1.1 DeepSeek-V3.2: DeepSeek Sparse Attention (DSA) và *lightning indexer* [V]

- Bài: *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, arXiv
  [2512.02556](https://arxiv.org/abs/2512.02556). Kiến trúc giống V3.1-Terminus; thay đổi duy nhất
  là DSA, đưa vào bằng continued training.
- **Lightning indexer** (Eq. 1): `I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)`, gồm H^I head
  nhỏ, có thể chạy FP8. Mỗi query token chỉ giữ **top-k = 2048** token KV cho attention chính.
- **Huấn luyện indexer** (gần với cách nhóm có thể huấn luyện "bộ chấm điểm"):
  - *Dense warm-up*: **đóng băng toàn bộ mô hình, chỉ huấn luyện indexer**, loss
    `Σ_t KL(p_t,: || softmax(I_t,:))`, trong đó `p` là attention chính cộng qua các head rồi chuẩn
    hoá L1. Chạy 1.000 bước × 16 chuỗi × 128K token, khoảng 2,1B token.
  - *Sparse training*: 15.000 bước, khoảng 943,7B token; đầu vào indexer được `detach` khỏi đồ thị
    tính toán để tối ưu riêng.
- Ý nghĩa: indexer là **"trọng số riêng để truy hồi"** đúng nghĩa: một mô hình chấm điểm nhỏ, học
  bằng cách chưng cất phân bố attention của mô hình lớn, rồi chọn cứng top-k.

### 1.2 DeepSeek-V4 (Pro/Flash): CSA + HCA [V-tóm]

- *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, arXiv
  [2606.19348](https://arxiv.org/abs/2606.19348) (nộp 26/04/2026). V4-Pro có 1.6T tham số (49B
  kích hoạt), V4-Flash có 284B (13B kích hoạt), ngữ cảnh 1M. Tác giả báo cáo V4-Pro cần khoảng 27%
  FLOPs/token và khoảng 10% KV cache so với V3.2 ở 1M token.
- **Compressed Sparse Attention (CSA)** (§2.3.1): mỗi nhóm m token được gộp thành 1 mục KV bằng
  *softmax-weighted pooling* có trọng số học được (các cửa sổ chồng lấn), sau đó **lightning
  indexer** chọn top-k mục *đã nén*.
- **Heavily Compressed Attention (HCA)** (§2.3.2): nén mạnh hơn (m' ≫ m), không có indexer,
  attention dày trên toàn bộ mục đã nén. Cả hai đều có thêm nhánh sliding-window cho các token gần,
  và được xếp xen kẽ theo lớp.
- Các giá trị m, m', k không đọc được trong phần đã truy cập (UNVERIFIED). Việc V4 dùng Engram:
  UNVERIFIED (100k ký tự đầu của bài không nhắc tới).
- **Liên hệ với RAG của nhóm:** HCA tương ứng với "mọi tài liệu đều được nén thành vài vector";
  CSA tương ứng với "indexer chọn vài mục để xem ở độ phân giải cao". Đây chính là motivation cho
  biến thể **lai** ở §5 (giữ top-1/2 tài liệu dạng text, phần còn lại dạng vector nhớ). REFRAG đã
  làm một phiên bản dùng RL (§2), nên đây không phải ý tưởng mới tuyệt đối.

### 1.3 DeepSeek-V4.1-Flash (09/2026) [V-tóm, đọc §2.2–2.4 của bản HTML]

- *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*, arXiv
  [2609.19969](https://arxiv.org/abs/2609.19969). MoE đa phương thức 552B, ngữ cảnh 1M, **8B tham
  số kích hoạt khi prefill và 16B khi decode**. KV cache trong HBM là 890 byte/token, khoảng ¼ của
  V4-Flash. Checkpoint và tech report có trên
  [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). DeepSeek
  [thông báo](https://api-docs.deepseek.com/news/news260910/) V4-Flash đã ngừng và chuyển sang
  V4.1-Flash.
- **Causal Encoder-Decoder (CED)** (§2.2): 40 lớp chia thành 20 lớp encoder nhân quả và 20 lớp
  decoder. **KV toàn cục của decoder không tính từ hidden state của decoder**, mà được chiếu (Eq. 1,
  trọng số `W^KV_l`, `W^Z_l` riêng cho từng lớp) từ đầu ra **lớp giữa của encoder** `H_{L/2}`. Vì
  vậy khi prefill chỉ cần chạy encoder trên prompt, và đó là lý do số tham số kích hoạt khi prefill
  (8B) bằng một nửa khi decode (16B). Cách giải thích "prefill chỉ chạy encoder" là suy luận của
  bản tóm tắt, bài không nói thẳng. Bài trích dẫn YOCO
  ([2405.05254](https://arxiv.org/abs/2405.05254)) làm tiền thân.
  → **Đây là cấu trúc gần ý tưởng của bạn nhất**: encoder chạy một lần, sinh "bộ nhớ" (KV), decoder
  đọc bộ nhớ đó. Khác biệt: nó được *tiền huấn luyện từ đầu* trên 45T token, nằm trong một mô hình,
  và bộ nhớ là KV từng token (nén m=2), không phải vài vector cho mỗi tài liệu.
- **CSA2** (§2.3): encoder nén m=2, decoder m=1. **Indexer** (32 head, dim 128) chấm điểm các mục KV,
  mỗi query giữ **top-k = 512** mục cộng với SWA cục bộ. Các lớp dùng chung KV/indexer theo ba chế
  độ Full / Reindex / Reuse. **Hierarchical Sparse Indexer** (chỉ ở decoder): lớp Full đầu tiên chọn
  2.048 khối × 8 vị trí (pool 16.384 vị trí), các lớp Reindex sau chỉ tìm trong pool đó. Đây là
  truy hồi hai tầng (thô → tinh), giống pipeline retriever → reranker.
- **Engram** (§2.4.2): 196B tham số ở lớp 1 và 14, n-gram bậc {2,3,4}, 8 hash head, khoảng 16M mục
  mỗi head; tra cứu tất định nên có thể prefetch từ RAM host. Đây là **bộ nhớ tham số tĩnh tra theo
  n-gram**, không truy hồi trên ngữ cảnh.

### 1.4 Engram [V]

- *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs*, arXiv
  [2601.07372](https://arxiv.org/abs/2601.07372), mã [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)
  (Apache-2.0). Hiện đại hoá embedding N-gram cổ điển để tra cứu O(1), xem như một "trục thưa" bên
  cạnh MoE. Tác giả báo cáo luật scaling hình chữ U khi phân bổ tham số giữa MoE và Engram. Engram-27B
  thắng MoE cùng số tham số và FLOPs: MMLU +3.4, CMMLU +4.0, BBH +5.0, ARC-C +3.7, HumanEval +3.0,
  MATH +2.4, **Multi-Query NIAH 84.2 → 97.0**. Lý giải của họ: Engram gánh việc "tái tạo tri thức
  tĩnh" cho các lớp đầu, nhờ đó attention có thêm dung lượng cho ngữ cảnh toàn cục.

### 1.5 Kết luận cho nhóm

| Thứ bạn nhớ | Cơ chế thật | Tương đương trong RAG của nhóm |
|---|---|---|
| "Trọng số riêng cho truy hồi" | Lightning indexer / Hierarchical Sparse Indexer: scorer nhỏ, học bằng KL theo attention, chọn top-k | Reranker/pruner top-k (Track A), hoặc indexer chưng cất từ attention của reader (Sentinel, AttnComp, ICR, QRHead đã làm) |
| "Encoder nén một lần, decoder đọc" | CED (V4.1-Flash): KV toàn cục của decoder chiếu từ encoder | COCOM/PISCO/OSCAR/xRAG (vector nhớ cho mỗi tài liệu), CEPE (encoder + cross-attention) |
| "Ma trận tham số" | Engram (bảng embedding n-gram), Cartridges (KV học riêng cho từng corpus) | Parametric RAG / DyPRAG (tài liệu → LoRA), Cartridges |

Không nên viết rằng nhóm "theo DeepSeek". Nên dùng DeepSeek làm **động lực và related work**:
hướng công nghiệp đang tách (i) bộ chọn lọc rẻ và (ii) bộ nhớ nén khỏi phần tính toán chính.

---

## 2. Ý tưởng của bạn trong bản đồ nghiên cứu

### 2.1 Phân loại

| Nhánh | "Ma trận" là gì | Bài tiêu biểu | Ghi chú |
|---|---|---|---|
| **Vector nhớ (soft tokens) cho mỗi đoạn** | k vector trong không gian embedding của decoder | ICAE [2307.06945](https://arxiv.org/abs/2307.06945) (ICLR'24), Gist [2304.08467](https://arxiv.org/abs/2304.08467), AutoCompressor [2305.14788](https://arxiv.org/abs/2305.14788), xRAG [2405.13792](https://arxiv.org/abs/2405.13792) (NeurIPS'24), COCOM [2407.09252](https://arxiv.org/abs/2407.09252), 500xCompressor [2408.03094](https://arxiv.org/abs/2408.03094), PISCO [2501.16075](https://arxiv.org/abs/2501.16075), OSCAR [2504.07109](https://arxiv.org/abs/2504.07109), REFRAG [2509.01092](https://arxiv.org/abs/2509.01092), No Mean Feat [2510.20797](https://arxiv.org/abs/2510.20797), CLaRa [2511.18659](https://arxiv.org/abs/2511.18659) | **Đúng ý tưởng của bạn**; có checkpoint |
| Vector nhớ, thế hệ 2026 | như trên, thêm chọn lọc hoặc phân bổ | SeleCom [2602.15856](https://arxiv.org/abs/2602.15856) (WWW'26), Density-aware [2603.25926](https://arxiv.org/abs/2603.25926), EnComp [2603.09222](https://arxiv.org/abs/2603.09222) (AACL'26), Meta-Soft [2605.22337](https://arxiv.org/abs/2605.22337), DEX-Comp [2609.05152](https://arxiv.org/abs/2609.05152), AdaMem [2609.22100](https://arxiv.org/abs/2609.22100) | Xu hướng: query-conditioned, phân bổ ngân sách theo độ liên quan; chưa thấy checkpoint |
| KV tính sẵn cho mỗi đoạn | KV cache của từng tài liệu | TurboRAG [2410.07590](https://arxiv.org/abs/2410.07590), Block-Attention [2409.15355](https://arxiv.org/abs/2409.15355) (ICLR'25), CacheBlend [2405.16444](https://arxiv.org/abs/2405.16444), KVLink [2502.16002](https://arxiv.org/abs/2502.16002), KV-Distill [2503.10337](https://arxiv.org/abs/2503.10337) | "Encode once" nhưng không giảm số vị trí (trừ KV-Distill) |
| **Tham số** học riêng cho từng corpus/tài liệu | prefix KV hoặc LoRA | Cartridges [2506.06266](https://arxiv.org/abs/2506.06266), LLoCO [2404.07979](https://arxiv.org/abs/2404.07979) (EMNLP'24), Parametric RAG [2501.15915](https://arxiv.org/abs/2501.15915), DyPRAG [2503.23895](https://arxiv.org/abs/2503.23895) | Đúng nghĩa đen "ma trận tham số"; hợp với corpus cố định, không hợp với 20 tài liệu thay đổi theo từng câu hỏi |
| Kiến trúc encoder/decoder | cross-attention hoặc KV chiếu từ encoder | RETRO [2112.04426](https://arxiv.org/abs/2112.04426), CEPE [2402.16617](https://arxiv.org/abs/2402.16617) (ACL'24), YOCO [2405.05254](https://arxiv.org/abs/2405.05254), DeepSeek CED, Memory Layers [2412.09764](https://arxiv.org/abs/2412.09764), Engram | Cần tiền huấn luyện lớn → ngoài tầm |
| Attention làm "indexer" | điểm attention / probe | ICR [2410.02642](https://arxiv.org/abs/2410.02642), QRHead [2506.09944](https://arxiv.org/abs/2506.09944), Sentinel [2505.23277](https://arxiv.org/abs/2505.23277), AttnComp [2509.17486](https://arxiv.org/abs/2509.17486), DSA | Là nén cứng; đã có nhiều bài |

### 2.2 Số liệu công bố (thiết lập khác LITM; chỉ để thảo luận)

| Bài (bảng) | Decoder | Thiết lập | Không nén | Nén mềm | Tỉ lệ | Trạng thái |
|---|---|---|---|---|---|---|
| xRAG (Table 1) | Mistral-7B, **đóng băng**, chỉ huấn luyện projector MLP | 1 tài liệu, EM | NQ 42.71 (no-retrieval 30.25) | **39.10** | ~175 token → 1 | [V-tóm] |
| COCOM (Table 2) | Mistral-7B-Instruct, tuned | 5 đoạn, EM | NQ 0.597 | 0.554 / 0.539 / 0.511 | 4x / 16x / 128x | [V-tóm] |
| COCOM (Table 6, 128x) | đóng băng so với tuned | 5 đoạn, EM | – | **0.421 (frozen) so với 0.519 (tuned)** | 128x | [V-tóm]; Table 2 ghi 0.511 còn Table 6 ghi 0.519 cho cùng cấu hình |
| PISCO (Table 2) | Mistral-7B, LoRA | 5 đoạn, SPLADE+DeBERTa rerank, accuracy ("match") | NQ 0.69 | **0.65** / 0.61 | 16x / 128x | [V-tóm] |
| PISCO | Llama-3.1-8B | như trên | NQ 0.65 | 0.64 | 16x | [V-tóm] |
| PISCO | Solar-10.7B | như trên | NQ 0.71 | 0.70 | 16x | [V-tóm] |
| OSCAR | Mistral-7B, LoRA; compressor Llama-3.2-1B | 10 đoạn (huấn luyện với 5), LLM-judge | NQ 0.68 | **0.68** (Provence 0.69, PISCO 0.65) | 16x | [V-tóm] |
| CLaRa (Table 1, compressor) | Mistral-7B | CEM | NQ 54.58 | SCP 57.05 / 55.56 / 53.36; PISCO 54.39; xRAG 32.35 | 4x / 16x / 128x | [V-tóm] |
| 500xCompressor (Table 5) | LLaMA-3-8B **đóng băng**, LoRA ở encoder, truyền KV | ArxivQA + 4 bộ, F1 giữ lại | 100% | 72.9% / 67.1% / 62.3% | 500→16 / 4 / 1 | [V-tóm]; không có NQ |
| No Mean Feat | Qwen3-8B | SQuAD / HotpotQA F1 | 86.48 / 84.67 | mean-pool 83.76 / 83.30 (4x); 44.65 / 63.88 (128x) | 4x, 128x | [V-tóm]; không có NQ |
| REFRAG | LLaMA-2-7B + RoBERTa-large | 16 tác vụ | – | khớp LLaMA với 10 đoạn, TTFT ×5.26; ×30.85 ở k=32 (so với CEPE) | 8–32 | [V-tóm]; code **chưa phát hành** |
| Cartridges | – | long-context | ICL | khớp ICL, bộ nhớ ít hơn 38.6x, throughput ×26.4 | – | [V], chỉ đọc abstract; base model UNVERIFIED |

Tốc độ (PISCO Table 3, A100, 5 tài liệu × 128 token): Mistral-7B 0.26 s và 11.231 GFLOPs, so với
PISCO-16x 0.06 s và 2.803 GFLOPs. Bài ghi lúc "5.7x", lúc "4x" (mâu thuẫn trong chính bài). OSCAR
(Mistral-7B, *tính cả thời gian nén*): 141.6 ms so với 50.1 ms (khoảng 2.8x).

Chi phí huấn luyện công bố: PISCO dùng 453k câu hỏi, 1×A100, dưới 48 h cho 7–10B (LoRA r=16,
lr 1e-4, batch 128, 1 epoch). OSCAR dùng khoảng 893k câu hỏi, 1–5 GPU-ngày. xRAG dùng khoảng 955k
mẫu trên 8×A100. COCOM fine-tune 493k QA và pretrain khoảng 10M đoạn. **Ngân sách của nhóm (khoảng
15k ví dụ, một đêm trên 4070S) nhỏ hơn khoảng 30 lần** → phải coi đây là "budget recipe" và nói rõ
trong báo cáo.

Thiếu số NQ cụ thể (UNVERIFIED): DEX-Comp (tác giả báo cáo vượt full-context chưa tune ở 16x, TTFT
giảm 4.4–23.7x), AdaMem (+3.2 điểm so với phân bổ đều ở 16x), SeleCom (giảm 33.8–84.6% tính toán).
Backbone của ba bài này chưa đọc được.

### 2.3 Bằng chứng nói gì về tiền đề "nén cực mạnh mà vẫn chính xác cao"

- Decoder **được fine-tune** + nén 16x: mất 0–4 điểm trên NQ (PISCO, COCOM); OSCAR và CLaRa ngang
  hoặc nhỉnh hơn baseline của chính họ. Lên 128x thì mất thêm vài điểm.
- Decoder **đóng băng** (chỉ huấn luyện encoder/projector): mất khoảng 10 điểm EM (COCOM Table 6),
  xRAG 1-token mất 3.6 EM dù dùng khoảng 1M mẫu và 8×A100.
- Cái lợi đo được là **FLOPs/TTFT/bộ nhớ** (3–5x end-to-end, hơn nữa nếu nén offline), không phải
  độ chính xác.
- **Hạn chế cốt lõi cho dự án:** nén mềm **không dùng được với API thương mại**, vì cần đưa embedding
  vào thẳng mô hình. Nó cũng gắn với *một* decoder cụ thể. Đây là điểm đối lập rõ với nén cứng
  (Provence…) và nên được viết thành một mục thảo luận trong báo cáo.

### 2.4 Khoảng trống phù hợp với nhóm

1. **Vị trí.** Chúng tôi không tìm thấy bài nào đo nén mềm theo giao thức vị trí của LITM. PISCO và
   OSCAR huấn luyện với k=5 tài liệu; OSCAR đánh giá với k=10. Với 20 tài liệu (160 vector nhớ) mà
   tài liệu vàng ở giữa thì sao? Nén từng tài liệu độc lập có làm phẳng đường cong chữ U như
   LongLLMLingua đã làm với nén cứng không?
2. **Cứng và mềm trên cùng một bộ.** OSCAR có so với Provence (0.68 so với 0.69), nhưng chưa ai so
   trên NQ-20 theo vị trí cùng với LLMLingua-2/LongLLMLingua.
3. **Công thức ngân sách thấp.** Mọi bài đều dùng A100/H100 và trên 450k mẫu. Một công thức 12 GB,
   khoảng 15k mẫu, mô hình 1.5B có số đo trung thực là một đóng góp thực tế nhỏ nhưng thật.
4. **Lai cứng→mềm** (Provence/reranker → nén mềm, hoặc top-1 giữ dạng text + phần còn lại dạng
   vector). REFRAG làm bằng RL với LLaMA-2; chưa thấy phiên bản đơn giản dùng reranker có sẵn.

---

## 3. Kiểm tra khả thi đã làm (không dùng GPU)

| Hạng mục | Kết quả | Hệ quả |
|---|---|---|
| Ổ đĩa | `~` (chứa `~/.cache/huggingface`) **còn 4.9 GB (96%)**; `/mnt/d` còn 605 GB | **Bước đầu tiên:** đặt `HF_HOME=/mnt/d/hf` (I/O qua 9p chậm hơn nhưng đủ dùng). Mistral-7B bf16 khoảng 14.5 GB, Qwen2-7B khoảng 15 GB |
| Độ dài tài liệu LITM (tokenizer Qwen2.5, 500 câu × 20 tài liệu) | title+text: trung bình 147, p50 146, p95 177, max 320; **89% vượt 128 token** | PISCO/OSCAR cắt mỗi tài liệu ở 128 token |
| Đáp án còn trong 128 token đầu của tài liệu vàng (n=500, gold@0) | text-only 0.992, title+text 0.986, 256 token 1.000 | Cắt ở 128 token chỉ mất khoảng 1% → chấp nhận, ghi chú. Tỉ lệ thực tế khoảng 147 → 8 vector ≈ **18x** |
| Gated model | `meta-llama/Llama-3.2-1B-Instruct`: **gated=manual**; Mistral-7B-Instruct-v0.2, Qwen2/2.5: không gated | OSCAR-Qwen2 cần Llama-1B (compressor gọi `AutoModel.from_pretrained(compr_model_name)`). Gửi yêu cầu truy cập ngay, hoặc sửa `config.json` *bản local* trỏ tới mirror `unsloth/Llama-3.2-1B-Instruct` (không gated, cùng trọng số; vẫn chịu license Llama 3.2). **PISCO-mistral không vướng gate → chọn làm checkpoint chính** |
| Mã nguồn `modelling_pisco.py` (đọc trong repo HF của OSCAR) | `AutoModel` + `trust_remote_code`; mặc định `attn_implementation='flash_attention_2'`; `quantization` ('no'/'int4'/'int8' bằng bitsandbytes) đọc từ **config**, `from_pretrained` không ghi đè được | Truyền `attn_implementation='sdpa'`; để chạy 4-bit thì sửa `config.json` bản local: `"quantization": "int4"`. Cần `uv add --group dev peft bitsandbytes`. Mã viết cho transformers 4.48, repo đang dùng 4.57.6 → smoke test trước. Ghim `revision` (OSCAR `d97ea5362e…`, PISCO `2121d0c0e8…`) |
| Kích thước checkpoint | pisco-mistral: adapters 168 MB + first/last layers 525 MB (cộng base Mistral-7B); oscar-qwen2-7B: adapters 81 MB + compressor 2.47 GB + mlp 91 MB + first/last 2.17 GB (cộng base Qwen2-7B) | Bộ nhớ GPU ước tính ở int4: decoder nf4 khoảng 5 GB (+ embedding/lm_head bf16) + compressor 1B khoảng 2.5 GB → **khoảng 7–9 GB**, vừa 12 GB khi vLLM đã tắt |
| vLLM | Không có trong venv; báo cáo ghi "vLLM 0.31" phục vụ `Qwen2.5-7B-Instruct-AWQ` | `--enable-prompt-embeds` (Completions và Chat Completions, trộn text với embedding) có trong [docs vLLM](https://docs.vllm.ai/en/latest/features/prompt_embeds.html), nhưng tổ hợp AWQ + LoRA + prompt_embeds **chưa kiểm chứng** → chỉ để làm demo tuỳ chọn; **đường đánh giá chính là HF transformers** |
| Dữ liệu huấn luyện | [`Tevatron/wikipedia-nq`](https://huggingface.co/datasets/Tevatron/wikipedia-nq): train 58.622 câu, có `positive_passages` + `negative_passages` (đoạn DPR-Wikipedia 100 từ, *cùng kiểu đoạn với LITM*); file `nq-train.jsonl.gz` khoảng 1.3 GB | Tách biệt NQ-train với 2.655 câu LITM là **UNVERIFIED** (chưa xác nhận LITM lấy split nào của NQ-Open) → Ngày 1: chuẩn hoá câu hỏi, đếm trùng với *toàn bộ* 2.655 câu LITM rồi loại các câu trùng. Kiểm tra mỗi câu có ≥19 negative (cho k=20); nếu thiếu thì mượn negative của câu hỏi train khác. **Không** lấy đoạn nhiễu từ corpus gộp 50.7k của dự án (corpus này chứa đoạn của tập test) |

---

## 4. Các phương án, chi phí và rủi ro

Phần cứng: RTX 4070 SUPER 12 GB. Đỉnh bf16 tensor (tích luỹ FP32) khoảng 70 TFLOPS, *suy ra từ
spec, UNVERIFIED*; HF + LoRA thực tế khoảng 15–25 TFLOPS. GPU **không chạy song song** vLLM và
huấn luyện → chỉ có khoảng 4 "đêm GPU" (19–22/10).

Ước lượng FLOPs/ví dụ khi huấn luyện LoRA, có gradient checkpointing: ≈ 6·N·T (N là số tham số
không tính embedding, T là số token đi qua mô hình cần lan truyền ngược). Mỗi tài liệu khoảng
128+8 token ở phía encoder; decoder khoảng k·9 + 70 (prompt) + 10 (đáp án) vị trí.
**Mọi con số dưới đây là khoảng ước lượng, phải đo lại bằng 10 bước smoke ở Ngày 1.**

| # | Phương án | Đóng góp | GPU | Thời gian ước tính | Rủi ro | Kết luận |
|---|---|---|---|---|---|---|
| **P0** | Chạy **checkpoint có sẵn** PISCO-Mistral (+ OSCAR-Qwen2 nếu có Llama) trên LITM NQ-20, vị trí {0,9,19}, kèm baseline full-text, closed-book, oracle của *chính* decoder đó với *cùng* prompt; thêm lai "reranker top-5 → PISCO" | Đo vị trí cho nén mềm (khoảng trống 1, 2, 4) | int4 khoảng 7–9 GB | Nén 10k tài liệu (query-independent: một lần cho cả 5 vị trí): PISCO nén bằng *chính decoder 7B nf4* (adapter encoder) nên mất **khoảng vài chục phút**; compressor 1B của OSCAR chỉ mất vài phút. Sinh câu trả lời 500×3 với decoder 7B nf4, đầu vào khoảng 250 vị trí: khoảng 15–30 phút. Full-text Mistral 1.500 × 2.9k token: khoảng 30–60 phút | Mã remote cũ hơn transformers đang dùng; nf4 làm lệch LoRA chút ít | **Làm chắc chắn** |
| **P1** | **Tự huấn luyện** compressor "encode-once" kiểu PISCO trên **Qwen2.5-1.5B-Instruct**: cùng trọng số làm encoder và decoder, 2 adapter LoRA, 8 mem token cho mỗi 128 token, prompt = `build_prompt` của LITM với text tài liệu thay bằng slot nhớ | Ý tưởng của bạn + khoảng trống 3; CPU demo được | bf16 khoảng 3 GB + activation → khoảng 6–9 GB | k=10: encoder khoảng 1.400 token, 6·1.3e9·1.4e3 ≈ 1.1e13 FLOPs + decoder ≈ 1.3e12 → **khoảng 0.5–0.8 s/ví dụ → 15k ví dụ ≈ 2–3.5 h**. k=20: khoảng gấp đôi | Dữ liệu ít hơn khoảng 30 lần so với bài gốc; reader 1.5B yếu hơn 7B | **Đóng góp chính, có cổng dừng** |
| P1' | Như P1 nhưng decoder **Qwen2.5-7B-Instruct QLoRA nf4**, encoder = 8 lớp đầu (OSCAR-N-layers) | Cùng họ reader với bảng chính | nf4 khoảng 5.5 GB + activation → khoảng 9–11 GB, sát giới hạn | k=5: khoảng 1.2e13 FLOPs/ví dụ, nf4 chậm → **khoảng 1–2 s/ví dụ → 15k ≈ 4–8 h**; k=20 thì không vừa một đêm | Chậm, không chạy được trên CPU, vẫn phải chạy lại full-text cho reader nf4+LoRA | Chỉ làm nếu P1 xong sớm |
| P2 | Reader **đóng băng** Qwen2.5-7B, chỉ huấn luyện encoder 0.5B + MLP chiếu sang 3584-d (kiểu xRAG/500x), phục vụ qua vLLM `prompt_embeds` | "Plug-in cho reader đang dùng" | lan truyền ngược qua 7B nf4: khoảng 9 GB | khoảng 0.5–1 s/ví dụ → 15k ≈ 2–4 h | Bằng chứng COCOM/xRAG: decoder đóng băng mất khoảng 10 điểm *ngay cả với* khoảng 1M mẫu → với 15k mẫu nhiều khả năng yếu; tổ hợp vLLM AWQ + prompt_embeds chưa kiểm chứng | Chỉ làm ablation/stretch |
| P3 | Indexer kiểu DSA: scorer nhỏ chưng cất bằng KL theo attention của reader để chọn tài liệu/câu | Gắn trực tiếp với DeepSeek | – | 1–2 ngày | Trùng với Sentinel/AttnComp/ICR/QRHead; vẫn là nén cứng | Không khuyến nghị; nhắc trong related work |
| P4 | Cartridges / Parametric RAG (huấn luyện tham số riêng cho từng tài liệu) | "Ma trận tham số" đúng nghĩa đen | – | Phải huấn luyện riêng cho mỗi câu hỏi/tài liệu | Không hợp với 20 tài liệu đổi theo từng câu hỏi | Loại |

Vì sao chọn **1.5B** cho P1 (không chọn 7B):
- Huấn luyện nhanh hơn khoảng 4 lần, đủ một đêm GPU.
- Đáp ứng yêu cầu "chạy không cần GPU": với nén **offline (query-independent)**, bộ nhớ của corpus
  được tính sẵn. Lúc hỏi, decoder chỉ xử lý khoảng 250 vị trí thay vì khoảng 2.9k token, tức khoảng
  2·1.5e9·250 ≈ 0.75 TFLOPs so với khoảng 8.7 TFLOPs. Trên CPU 8 luồng đây là chênh lệch vài giây
  so với vài chục giây (*cần đo*). Đây là demo trực quan nhất cho câu "encode once".
- Dù chọn reader nào thì cũng phải chạy dòng full-text riêng cho reader đó (LoRA hoặc nf4 đã là một
  mô hình khác về số so với AWQ của bảng chính). Vì vậy "cùng họ Qwen 7B" không mua được sự công
  bằng tuyệt đối.

Thiết kế P1 chi tiết (đề xuất):
- **Mã nền:** lớp `COCOM` trong `modelling_pisco.py` (repo HF của naver, cùng code với BERGEN
  [`naver/bergen`](https://github.com/naver/bergen) `models/generators/llm_pisco.py`, license
  CC BY-NC 4.0, đủ cho mục đích học thuật) với `decoder_model_name="Qwen/Qwen2.5-1.5B-Instruct"`,
  `compr_model_name=None` (decoder kiêm encoder), `compr_rate=16`, `doc_max_length=128`, `lora_r=16`,
  `training_form="both_separately"`, `attn_implementation="sdpa"`. Một phương án thay thế *đơn giản
  hơn* là tự viết khoảng 200 dòng: encoder LoRA chạy hai chiều trên tài liệu → **mean-pool mỗi 16
  token** (No Mean Feat cho thấy mean-pool mạnh hơn mem-token; mã MIT
  [lil-lab/benchpress](https://github.com/lil-lab/benchpress), viết bằng JAX nên chỉ tham khảo).
  Chọn một trong hai sau smoke test. Lưu ý khi chọn: `blend_prompt_and_memory_tokens` của COCOM **hard-code prompt riêng** (system + "Background/Question"), nên muốn dùng `build_prompt()` thì phải override hàm này hoặc tự viết script. Hai điểm phải kiểm tra trong smoke test: (a) Qwen2.5-1.5B có `tie_word_embeddings=True`, trong khi đường lưu/nạp "first and last layers" của COCOM được viết cho Mistral (không tie) → kiểm tra khi thêm mem token; (b) số negative mỗi câu trong Tevatron đủ cho k=20.
- **Dữ liệu:** Tevatron NQ-train. Mỗi mẫu gồm 1 positive + (k−1) hard negative, **vị trí vàng ngẫu
  nhiên**, k lấy ngẫu nhiên trong {5, 10, 20}. Nhãn: đáp án ngắn gốc. Nếu còn thời gian thì dùng
  nhãn teacher (Qwen2.5-7B-AWQ qua vLLM với text đầy đủ), vì PISCO cho thấy chưng cất từ teacher
  tốt hơn nhãn vàng.
- **Prompt:** dùng đúng `build_prompt()` của LITM, chỉ thay phần text mỗi `Document [i](Title: …)`
  bằng 8 slot nhớ (giữ tiêu đề dạng text, rẻ và hữu ích). Như vậy reader 1.5B ở mọi dòng (full,
  closed-book, oracle, nén cứng, nén mềm) dùng cùng một khuôn prompt.
- **Biến thể (theo thứ tự ưu tiên):** (a) query-independent (encode-once); (b) **lai**: top-1/2 tài
  liệu theo reranker giữ dạng text, phần còn lại dạng nhớ. Trong lúc huấn luyện thì "mở" ngẫu nhiên
  0–2 tài liệu sang text; đây là phiên bản CSA/HCA của DeepSeek ở cấp RAG, REFRAG làm bằng RL;
  (c) query-aware (OSCAR-style: đưa câu hỏi vào encoder, mất tính encode-once).
- **Đối chứng quan trọng:** "LoRA-text": cùng 1.5B, cùng dữ liệu, fine-tune với text đầy đủ. Đối
  chứng này tách hiệu ứng *fine-tune trên NQ* khỏi hiệu ứng *nén*. Nếu không đủ GPU thì ghi rõ là
  thiếu đối chứng.

---

## 5. Đánh giá công bằng trên benchmark hiện có

- **Cùng dữ liệu:** `load_litm_nq(pos, n=500, seed=42)`, vị trí {0, 9, 19}. Nén mềm query-independent
  cho kết quả nén *giống nhau* ở mọi vị trí (chỉ khác thứ tự), giống Provence → nén một lần.
- **Cùng reader cho mỗi khối so sánh:**
  - Khối 7B-AWQ (bảng chính): giữ nguyên.
  - Khối Mistral-7B nf4 (P0): `closedbook`, `oracle`, `full` (text, prompt kiểu PISCO), `pisco`,
    `rerank_top5→pisco`. Prompt PISCO khác `build_prompt` (decoder được huấn luyện với prompt của
    nó) → **lệch khỏi AGENTS §4, phải ghi rõ**; chạy full-text bằng *cùng* prompt PISCO.
  - Khối Qwen2.5-1.5B (P1): `closedbook`, `oracle`, `full`, *LoRA-text*, cùng 2–3 nén cứng tốt nhất
    (dùng lại context đã nén trong `data/outputs/`, chỉ chạy lại reader), `soft16x`, `hybrid`.
  - So sánh **độ chênh so với full-text của chính reader** và **hình dạng đường cong theo vị trí**,
    đúng tinh thần §4 với số GPT-3.5.
- **Chỉ số và schema** (đề xuất; cần báo nhóm trước khi sửa `SUMMARY_FIELDS`):
  - `comp_tokens` = số **vị trí đầu vào decoder dành cho ngữ cảnh** (vector nhớ + tiêu đề), không đếm
    bằng tiktoken. `orig_tokens` vẫn đếm bằng `count_tokens` → `ratio` mang tính xấp xỉ, ghi chú
    trong báo cáo.
  - `compress_ms` = thời gian encoder. Với encode-once, báo cáo cả *amortized* (≈0 lúc hỏi) và *online*.
  - `llm_ms` đo trong HF (không phải vLLM) → so `llm_ms` giữa các dòng *trong cùng khối*.
  - `cost_usd_per_1k_q` = N/A (không có API nào nhận embedding).
  - `accuracy` = `best_subspan_em`, như mọi dòng khác. Đề xuất cột mới cuối danh sách:
    `reader` (vd. `mistral-7b-nf4`, `qwen2.5-1.5b`) và `ctx_positions`.
- **Độ mạnh thống kê:** với n=500, khoảng tin cậy 95% của mỗi ô là khoảng ±4 điểm. So sánh giữa các vị trí và giữa các method bằng **paired bootstrap** trên cùng 500 qid (cùng câu hỏi cho mọi method và vị trí). Chênh lệch dưới khoảng 4 điểm thì không kết luận.
- **Sanity check trước khi chạy LLM** (tương tự `--dry-run`): kiểm tra đáp án còn trong 128 token đầu
  của tài liệu vàng (đã làm: 98.6%); với lai thì đo `ctx_recall` của phần text giữ lại.
- **Báo cáo phải nêu** (các điểm trung thực): 20 tài liệu = 160 vector là *ngoài phân phối* huấn
  luyện của PISCO/OSCAR (k=5); P1 huấn luyện trên NQ còn nhiều nén cứng thì zero-shot (OpenProvence
  có thể đã thấy NQ, xem `docs/research.md` §4); nhiễu trong Tevatron là BM25, còn LITM là
  Contriever.

---

## 6. Kế hoạch theo ngày (đề xuất)

Phân công: P0 + P1 cộng thêm khoảng 1.5 người-ngày, trong khi Sơn đã giữ A0–A5 trong cùng các
ngày. Đề xuất **Hoàng chạy phần đánh giá P0** (chỉ là chạy script) và rút gọn danh sách model của
B2; Sơn lo P1 (dữ liệu, huấn luyện, cổng).

**Cắt khỏi `docs/plan.md` để có chỗ:** A3 LongLLMLingua (đã có số công bố để tham chiếu), A6,
B3 NanoNQ; XProvence/Zilliz chỉ giữ một.

| Ngày | Ban ngày (vLLM có thể chạy) | Đêm GPU (tắt vLLM) |
|---|---|---|
| **T2 19/10** | `HF_HOME=/mnt/d/hf`; `uv add --group dev peft bitsandbytes`; tải Mistral-7B-Instruct-v0.2 + `naver/pisco-mistral` (khoảng 15.5 GB); xin quyền Llama-3.2-1B (hoặc dùng mirror); tải Tevatron nq-train; `scripts/smoke_pisco.py` (int4, sdpa, 5 câu) | **P0:** PISCO 500×{0,9,19} + Mistral full / closed-book / oracle cùng prompt (khoảng 1.5–2 h) |
| **T3 20/10** | Viết `scripts/build_soft_train.py` (CPU) và `scripts/train_soft.py`; smoke 10 bước → **đo s/bước và VRAM thật**, chốt k và số mẫu | **P1 run 1:** Qwen2.5-1.5B, khoảng 15k mẫu, k∈{5,10,20}, vị trí ngẫu nhiên, hybrid-mixing bật (khoảng 3–5 h) |
| **T4 21/10** | **Cổng:** đánh giá 200 câu Tevatron-dev (không phải LITM). Nếu accuracy > closed-book + 10 điểm và ≥ 70% full-text của 1.5B thì đi tiếp; nếu không thì **dừng P1**, chỉ dùng P0 | Nếu qua cổng: LITM 500×3 cho soft16x / hybrid + các dòng của reader 1.5B (full, closed, oracle, 2–3 nén cứng). Nếu dư: run 2 = LoRA-text hoặc query-aware |
| **T5 22/10** | P0 lai (OSCAR nếu có Llama); đo latency CPU encode-once so với full-text; tab demo "soft memory" (CPU) | (dự phòng) chạy lại / full 2.655 câu cho 1 cấu hình |
| T6–CN 23–25/10 | Viết báo cáo theo `docs/plan.md` (thêm mục "Soft compression"), slide | – |

**Phương án dự phòng (theo mức):**
1. P1 không qua cổng → báo cáo P0 (vị trí, cứng so với mềm, lai) cộng *phân tích thất bại* của P1
   (đường loss, accuracy trên dev; vì sao 15k mẫu chưa đủ, đối chiếu với 453k của PISCO). Đây vẫn là
   kết quả trung thực, có giá trị.
2. Checkpoint có sẵn không load được (xung đột phiên bản) → tạo venv riêng `transformers==4.48`
   cho đúng một script. Nếu vẫn hỏng thì chỉ trích số công bố (§2.2, ghi "different setting") và
   đưa thiết kế vào phần "hướng phát triển".

---

## 7. Rủi ro, license, an toàn

- `trust_remote_code` (PISCO/OSCAR): đã đọc lướt `modelling_pisco.py` (1.113 dòng): chỉ dùng
  transformers/peft/bitsandbytes, `torch.load(weights_only=True)`, `hf_hub_download`. Vẫn **ghim
  revision**.
- License: PISCO/OSCAR có header HF ghi `cc-by-4.0` nhưng thân model card ghi **CC BY-NC 4.0**, tức
  mâu thuẫn → coi là phi thương mại (học thuật OK). Llama-3.2 dùng Community License. Mistral-7B-v0.2
  và Qwen2/2.5-1.5B/7B dùng Apache-2.0. Tevatron NQ: license chưa ghi trên HF (UNVERIFIED; NQ gốc là
  CC BY-SA 3.0).
- Không commit weights/adapter (`*.pth`, `*.safetensors`) theo AGENTS §6; adapter tự huấn luyện để
  dưới `data/` hoặc `/mnt/d`.
- Mâu thuẫn trong nguồn (ghi lại, không tự giải quyết): PISCO tốc độ "5.7x" so với "4x"; COCOM 128x
  0.511 (Table 2) so với 0.519 (Table 6); REFRAG ×30.85 so với ×32.99 và "không mất perplexity" là
  so với CEPE, không phải full-context; CLaRa F1 50.89 (bảng) so với 51.41 (văn bản).

---

## 8. Nguồn chính (đã mở trực tiếp)

- DeepSeek-V3.2 — https://arxiv.org/abs/2512.02556 (HTML: §DSA, Eq. 1, warm-up/sparse training) [V]
- DeepSeek-V4 — https://arxiv.org/abs/2606.19348 (§2.3.1 CSA, §2.3.2 HCA) [V-tóm]
- DeepSeek-V4.1-Flash — https://arxiv.org/abs/2609.19969 ; https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash ; https://api-docs.deepseek.com/news/news260910/ (§2.2 CED, §2.3 CSA2, §2.4.2 Engram) [V-tóm]
- Engram — https://arxiv.org/abs/2601.07372 ; https://github.com/deepseek-ai/Engram [V]
- PISCO — https://arxiv.org/abs/2501.16075 ; https://huggingface.co/naver/pisco-mistral [V-tóm]
- OSCAR — https://arxiv.org/abs/2504.07109 ; https://huggingface.co/naver/oscar-qwen2-7B (config.json, modelling_pisco.py đã đọc) [V]
- COCOM — https://arxiv.org/abs/2407.09252 ; checkpoint `naver/cocom-v1-{4,16,128}-mistral-7b` [V-tóm]
- xRAG — https://arxiv.org/abs/2405.13792 ; https://github.com/Hannibal046/xRAG ; `Hannibal046/xrag-7b` [V-tóm]
- CLaRa — https://arxiv.org/abs/2511.18659 ; https://github.com/apple/ml-clara ; `apple/CLaRa-7B-{Base,Instruct,E2E}` [V-tóm]
- REFRAG — https://arxiv.org/abs/2509.01092 (code "will be available" tại github.com/facebookresearch/refrag, hiện 404) [V-tóm]
- 500xCompressor — https://arxiv.org/abs/2408.03094 [V-tóm]
- No Mean Feat / BenchPress — https://arxiv.org/abs/2510.20797 ; https://github.com/lil-lab/benchpress (MIT) [V-tóm]
- Cartridges — https://arxiv.org/abs/2506.06266 ; https://github.com/HazyResearch/cartridges (Apache-2.0) [V, abstract]
- SeleCom 2602.15856, DEX-Comp 2609.05152, AdaMem 2609.22100 — chỉ đọc abstract [V, abstract]
- BERGEN (mã huấn luyện COCOM/PISCO, Provence) — https://github.com/naver/bergen
- vLLM prompt embeddings — https://docs.vllm.ai/en/latest/features/prompt_embeds.html
- Tevatron NQ — https://huggingface.co/datasets/Tevatron/wikipedia-nq
- Các ID khác (đã kiểm tra tiêu đề): ICAE 2307.06945, Gist 2304.08467, AutoCompressor 2305.14788,
  TurboRAG 2410.07590, Block-Attention 2409.15355, CacheBlend 2405.16444, KVLink 2502.16002,
  KV-Distill 2503.10337, LLoCO 2404.07979, Parametric RAG 2501.15915, DyPRAG 2503.23895,
  RETRO 2112.04426, CEPE 2402.16617, YOCO 2405.05254, Memory Layers 2412.09764, ICR 2410.02642,
  QRHead 2506.09944, Sentinel 2505.23277, AttnComp 2509.17486, Density-aware 2603.25926,
  EnComp 2603.09222, Meta-Soft 2605.22337.
