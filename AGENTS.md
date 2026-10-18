# AGENTS.md – Quy ước chung cho agent & thành viên

File này là "hợp đồng" để mọi agent (Claude Code, Codex, Cursor, …) của các thành viên làm việc
thống nhất. **Đọc hết trước khi sửa code.** `CLAUDE.md` chỉ import file này.

## 0. Quy trình nhận task – BẮT BUỘC trước khi làm bất cứ việc gì

`docs/tasks.md` là bảng task chung, cho biết ai đang làm gì và task nào còn trống. Mục đích là
để hai thành viên (và agent của họ) không làm trùng hoặc gây xung đột.

1. **Đồng bộ:** `git pull --rebase`, rồi đọc `docs/tasks.md`.
2. **Chọn task:**
   - Chỉ chọn task `todo` có các task phụ thuộc đã `done`.
   - **Không** sửa code/file thuộc phạm vi một task `doing` của người khác. Cần thì hỏi người đó
     hoặc thêm ghi chú vào cột "Cập nhật".
3. **Nhận task ngay:**
   - Điền "Người nhận" (vd. `Hoàng/Codex`), đổi trạng thái thành `doing`, ghi ngày giờ.
   - Commit chỉ thay đổi đó (`chore(tasks): claim T07`) và **push ngay**, trước khi bắt đầu làm.
   - Nếu push bị từ chối: `git pull --rebase`, kiểm tra lại. Nếu đã có người nhận task đó thì
     chọn task khác.
4. **Trong lúc làm:**
   - Commit nhỏ và **push thường xuyên**: sau mỗi bước chạy được, mỗi kết quả thí nghiệm mới,
     và tối thiểu khoảng mỗi giờ.
   - Cập nhật cột "Cập nhật" khi có tiến triển hoặc số liệu đáng chú ý.
   - Kết quả (`results/`) phải push ngay để người kia không phải chạy lại.
5. **Xong task:**
   - Chuyển task sang bảng "Đã xong", kèm 1 dòng kết quả chính và file liên quan.
   - Commit, push.
   - Mở khoá các task đang `blocked` vì task này (nếu có).
6. **Phát sinh việc mới:** thêm dòng với ID kế tiếp (`T16`, …), trạng thái `todo`. Không tự ý
   nhận việc lớn ngoài bảng mà chưa thống nhất với nhóm.
7. **Bị chặn hoặc bỏ dở:** đổi trạng thái thành `blocked` (ghi lý do) hoặc trả về `todo` (xoá tên
   người nhận) rồi push, để người khác nhận được.

## 1. Bối cảnh dự án

- BTL IT4772 NLP, đề tài *Efficient LLM/RAG*, dạng **lập trình**. Deadline **CN 25/10/2026**.
- Mục tiêu: benchmark **nén ngữ cảnh** (Track A) + **truy hồi hiệu quả** (Track B) cho RAG QA,
  so sánh với số liệu đã công bố, kèm demo Gradio. Xem `README.md`, `docs/research.md`,
  `docs/plan.md` (kế hoạch, phân công, ma trận thí nghiệm).
- Ưu tiên **dùng thư viện/checkpoint có sẵn**. Chỉ tự huấn luyện khi nhóm đã chốt (task T12 trong
  `docs/tasks.md`). Không mở rộng phạm vi ngoài `docs/tasks.md`/`docs/plan.md` nếu chưa thống
  nhất trong nhóm.

## 2. Môi trường

- Python **3.11** (`.python-version`), quản lý bằng **uv**. Không dùng `pip install` trực tiếp.
  - Cài: `uv sync` · Chạy: `uv run python ...` · Thêm thư viện: `uv add <pkg>`
    (dev/demo: `uv add --group dev|demo <pkg>`), commit cả `pyproject.toml` và `uv.lock`.
- Lần đầu: `cp .env.example .env`, điền khoá; `uv run python -c "import nltk; nltk.download('punkt_tab')"`;
  `uv run python scripts/download_data.py`.
- Đặt `HF_HOME` sang ổ còn trống (model ~10 GB). GPU có thể bận → mọi model phải chạy được trên
  CPU; ép thiết bị bằng `EFFRAG_DEVICE=cpu|cuda`.
- Kiểm tra trước khi commit: `uv run ruff check src scripts tests`, `uv run ruff format src scripts tests`,
  `uv run pytest -q`.

## 3. Cấu trúc & nơi đặt code

| Thứ cần thêm | Đặt ở đâu |
|---|---|
| Compressor mới | `src/effrag/compression/<name>.py`, kế thừa `Compressor`, cài `_compress()` + `params()`, đăng ký trong `get_compressor()` ở `compression/__init__.py` |
| Retriever mới | `src/effrag/retrieval.py` (hoặc tách `retrieval/` nếu lớn), interface `search(queries, k) -> list[list[int]]`, có thuộc tính `build_s` |
| Dataset mới | `src/effrag/data.py`, trả về `list[Example]` |
| Script thí nghiệm | `scripts/run_<track>.py`, ghi kết quả bằng `effrag.results` |
| Demo | `app/demo.py` (Gradio), chỉ gọi API của `effrag`, không copy logic |
| Notebook phân tích / vẽ hình | `notebooks/`, hình xuất ra `docs/report/figures/` |
| Báo cáo / slide | `docs/report/`, `docs/slide/` |

- Model nặng phải **import lazy** trong `__init__` của class (xem `llmlingua2.py`).
- Mỗi model mới: viết `scripts/smoke_<name>.py` chạy 1 mẫu trước khi tích hợp.
- Model dùng `trust_remote_code` (Provence, XProvence, OpenProvence, Zilliz) → **ghim `revision=`
  theo commit hash** khi đưa vào thí nghiệm chính thức. Khi chạy trong terminal tương tác có thể
  hiện hỏi `Do you wish to run the custom code? [y/N]` → trả lời `y`.
- **Không** load file `.pkl` lạ (vd. Sentinel) ngoài môi trường cách ly.

## 4. Giao thức thí nghiệm (BẮT BUỘC để số liệu so sánh được)

- **Dữ liệu chính:** `load_litm_nq(gold_pos, n=500, seed=42)` – cùng 500 câu hỏi cho mọi vị trí
  vàng `{0,9,19}` (đầu/giữa/cuối). Ma trận cấu hình nằm trong `scripts/run_all_compression.sh`:
  closed-book/oracle chỉ 1 vị trí (không phụ thuộc vị trí), LongLLMLingua 200 câu ở vị trí giữa.
- **Prompt:** dùng `build_prompt()` trong `generation.py` (prompt gốc của Lost in the Middle).
  Không sửa prompt cho riêng method nào.
- **Generator:** một `LLM_MODEL` duy nhất cho cả nhóm: `Qwen/Qwen2.5-7B-Instruct-AWQ` qua vLLM
  (served name `qwen2.5-7b-instruct`, `--max-model-len 4096`), `temperature=0`, `max_tokens=100`. Ghi tên
  model + ngày chạy (đã tự ghi vào `summary.csv`). Đổi model ⇒ chạy lại toàn bộ baseline.
- **Đếm token:** luôn dùng `effrag.tokens.count_tokens` (tiktoken `o200k_base`), không dùng
  tokenizer riêng của từng compressor.
- **Giữ nguyên thứ tự tài liệu/câu** sau khi nén (`reorder=False`), nếu không biến "vị trí vàng"
  mất ý nghĩa. Nếu thử sắp xếp lại thì báo cáo thành dòng riêng "reorder".
- **Ghép tỉ lệ nén:** mục tiêu 2x (~1.450 token) và 4x (~730 token) như bảng LongLLMLingua. Method
  dùng ngưỡng (Provence…) thì quét ngưỡng, báo cáo tỉ lệ thực tế đạt được.
- **Chỉ số:** `accuracy` = `best_subspan_em` (đáp án chuẩn hoá nằm trong output), kèm
  `ctx_recall`, `em`, `f1`, token, `ratio`, `compress_ms`, `llm_ms`, `cost_usd_per_1k_q`.
- Tiết kiệm tài nguyên: chỉ chạy một job nặng trên GPU tại một thời điểm (vLLM không chia GPU
  được với bộ nén), `EFFRAG_THREADS=8`. Cái gì đã có số công bố thì trích dẫn thay vì chạy lại.
- `transformers` ghim `<5` (OpenProvence và llmlingua chưa tương thích bản 5).
- Luôn chạy `--dry-run` (miễn phí) trước để kiểm tra `ctx_recall`/`ratio`, rồi mới gọi LLM.
- LLM được cache theo hash prompt trong `data/cache/llm/` → chạy lại không tốn tiền. Không xoá cache
  khi chưa cần.
- Số liệu công bố (GPT-3.5-Turbo-0613, đã ngừng) chỉ để **tham chiếu**: so sánh *độ chênh so với
  baseline của chính mình* và *hình dạng đường cong theo vị trí*, không so số tuyệt đối.

## 5. Schema kết quả

- `results/<track>/summary.csv`: 1 dòng / (run_id, method, params, gold_pos). Cột cố định trong
  `effrag.results.SUMMARY_FIELDS`. **Chỉ append, không sửa tay.** Thêm cột mới → sửa
  `SUMMARY_FIELDS` ở cuối danh sách và báo nhóm.
- `results/<track>/<run_id>_gold<pos>.jsonl`: từng câu (không chứa context) → commit.
- `data/outputs/…`: context sau nén để phân tích lỗi → không commit.
- Track: `compression`, `compression_dryrun`, `retrieval`, `e2e`.
- Run hỏng/thử nghiệm: xoá dòng tương ứng bằng script, ghi lý do trong commit message.

## 6. Git workflow

- Nhánh `main` luôn chạy được.
  - Thay đổi `docs/tasks.md`, kết quả (`results/`) và tài liệu thì commit **thẳng vào `main`** và
    push ngay, để mọi người thấy liền.
  - Thay đổi code lớn hoặc có rủi ro thì làm trên nhánh `feat/<tên-ngắn>` hoặc `exp/<tên>` →
    PR/merge sớm (trong ngày). Không giữ nhánh riêng lâu.
- Chạy `ruff check` trước khi push code.
- Commit nhỏ, message tiếng Anh dạng `feat: …`, `fix: …`, `exp: …`, `docs: …`.
- `git pull --rebase` trước khi push. Xung đột ở `summary.csv` → giữ cả hai phía (chỉ append).
- **Không bao giờ commit:** `.env`, API key, `data/`, weights model (`*.safetensors`, `*.bin`,
  `*.gguf`), `docs/org/` (PDF của giảng viên, mã số sinh viên).

## 7. Ngôn ngữ & phong cách

- Code, tên biến, docstring, commit: **tiếng Anh**. Tài liệu (`docs/`, README, báo cáo, slide):
  **tiếng Việt có dấu đầy đủ**.
- Ruff, line-length 100, type hints cho hàm public, docstring ngắn nêu nguồn bài báo (arXiv ID)
  cho mỗi method.
- Không bịa số liệu/trích dẫn: số trong báo cáo phải lấy từ `results/` hoặc ghi rõ bảng/bài báo
  gốc (xem `docs/research.md`). Chưa kiểm chứng ⇒ đánh dấu `UNVERIFIED`.

## 8. Ghi chép tiến độ

Tiến độ được ghi trong `docs/tasks.md` theo quy trình §0. Mỗi task xong có 1 dòng kết quả chính
(vd. "Provence τ=0.1: 4,6×, recall 0,916, n=500"); đây là nguồn để viết báo cáo.
`docs/plan.md` giữ kế hoạch/ma trận tổng thể, chỉ cập nhật khi đổi hướng.
