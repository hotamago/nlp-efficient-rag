# Kịch bản thuyết trình – Efficient RAG (15–20 phút)

Kịch bản đi kèm `docs/slide/slides.pdf` (18 trang). Tổng thời lượng dự kiến **≈ 18 phút**, còn
khoảng 2 phút dự phòng trong khung 20 phút. Mọi số liệu lấy từ slide/`results/`; khi nói chỉ cần
đọc **số in đậm**, phần còn lại để trên slide.

## Phân vai và thời gian

| Phần | Slide | Người nói (gợi ý) | Thời lượng | Mốc kết thúc |
|---|---|---|---|---|
| I. Bài toán & hệ thống | 1–5 | A | 4:20 | 4:20 |
| II. Phương pháp | 6–7 | A | 2:15 | 6:35 |
| III. Kết quả nén (Track A) | 8–13 | B | 7:00 | 13:35 |
| III. Truy hồi, end-to-end, CPU | 14–16 | B | 2:30 | 16:05 |
| IV. Demo & kết luận | 17–18 | A | 1:45 | 17:50 |

**Nếu bị chậm** (quá 10:00 khi mới sang slide 9): nói slide 12–13 trong 1 phút tổng, bỏ slide 15
(chỉ đọc câu "ý chính"), demo chỉ chỉ vào ảnh.

---

## Slide 1 – Trang bìa (0:30)

> Chúng em xin chào thầy cô và các bạn. Nhóm em gồm Nguyễn Hoàng Sơn và Võ Việt Hoàng, trình
> bày đề tài 13: **Efficient RAG – nén ngữ cảnh và truy hồi hiệu quả cho hỏi đáp với LLM**. Đây là
> đề tài dạng lập trình: nhóm xây một bộ benchmark, chạy lại các phương pháp nén và truy hồi trên
> cùng một giao thức, rồi so với số liệu đã công bố.

## Slide 2 – Nội dung (0:20)

> Bài trình bày gồm bốn phần: bài toán, phương pháp, kết quả, và tổng kết kèm demo.

*(Không đọc chi tiết, chuyển luôn.)*

## Slide 3 – Bài toán (1:30)

> RAG đưa các đoạn văn truy hồi được vào prompt. Với 20 đoạn của tập Natural Questions, prompt
> dài khoảng **2.900 token** chỉ để trả lời một câu hỏi vài chục token. Chi phí và độ trễ tăng theo
> độ dài đó.
>
> Vấn đề thứ hai là **lost in the middle**: bài báo của Liu và cộng sự năm 2024 cho thấy LLM dùng
> thông tin ở giữa ngữ cảnh kém hơn ở đầu và cuối, tạo thành đường cong chữ U.
>
> Có hai hướng giảm chi phí: **nén ngữ cảnh** – Track A – bỏ bớt token, câu hoặc đoạn thừa; và
> **truy hồi hiệu quả** – Track B – lấy đúng đoạn với chỉ mục nhỏ và nhanh.
>
> *[chỉ khung bên phải]* Nhóm đặt ra năm câu hỏi: nén 2–4 lần mất bao nhiêu độ chính xác; truy
> hồi đánh đổi thế nào giữa chất lượng, tốc độ, bộ nhớ; kết luận của bài báo – vốn dùng GPT-3.5 –
> còn đúng với LLM mở 7B không; không có GPU có làm được không; và cuối cùng là một hướng nhóm tự
> thử: nén mềm "encode-once".

## Slide 4 – Hệ thống (0:45)

> *[chỉ sơ đồ từ trái sang phải]* Pipeline gồm ba khâu: retriever lấy 20 đoạn, compressor nén lại,
> reader là **Qwen2.5-7B-Instruct** chạy trên vLLM sinh câu trả lời. Toàn bộ là một gói Python
> `effrag`: mỗi bộ nén, mỗi retriever cài cùng một interface, kết quả ghi chung một schema, nên
> thêm phương pháp mới không phải sửa script thí nghiệm. Câu trả lời LLM được cache theo hash
> prompt, chạy lại không tốn thêm.

## Slide 5 – Benchmark & giao thức (1:15)

> Dữ liệu là bộ NQ 20 đoạn của chính bài *Lost in the Middle*. Nhóm cố định **500 câu** với seed
> 42, và đặt đoạn chứa đáp án ở **vị trí 1, 10 và 20** – đúng ba điểm mà LongLLMLingua và
> LLMLingua-2 công bố số liệu, để so được.
>
> Chỉ số chính là accuracy: đáp án chuẩn hoá có nằm trong output hay không. Ngoài ra nhóm đo
> **context recall** – đáp án còn trong ngữ cảnh sau nén không – chỉ số này không cần gọi LLM.
> Mọi so sánh đều dùng paired bootstrap với khoảng tin cậy 95%.
>
> *[chỉ khung bên phải]* Vì tài nguyên hạn chế, nhóm thiết kế để tiết kiệm: chạy dry-run miễn phí
> trước, cache kết quả nén theo từng đoạn để dùng lại cho cả ba vị trí, và trích số đã công bố thay
> vì chạy lại.

## Slide 6 – Các phương pháp (1:00)

> Nhóm so sánh nén ở bốn mức. **Mức prompt**: cắt cụt, làm baseline. **Mức tài liệu**: BM25,
> dense, reranker chọn top-k đoạn. **Mức token**: LLMLingua-2 phân loại giữ/bỏ từng token, và
> LongLLMLingua dùng perplexity có điều kiện theo câu hỏi. **Mức câu**: Provence và OpenProvence
> cắt câu theo câu hỏi.
>
> Ngoài nén cứng còn có **nén thành vector**: PISCO của Naver, và mô hình encode-once nhóm tự
> huấn luyện. Tỉ lệ mục tiêu là **2 lần và 4 lần**, tức khoảng 1.450 và 730 token, như bảng của
> LongLLMLingua.

## Slide 7 – Đề xuất: nén mềm encode-once (1:15)

> Ý tưởng: nén cứng phải chạy lại **mỗi lần hỏi**. Nén mềm thì mã hoá kho tài liệu **một lần**,
> offline: mỗi đoạn tối đa 128 token được encoder biến thành **8 vector nhớ**. Lúc hỏi, decoder chỉ
> đọc các vector này thay cho văn bản – 20 đoạn còn khoảng 416 vị trí thay vì 2.900 token.
>
> Nhóm dùng **Qwen2.5-1.5B** với hai adapter LoRA trên cùng một model, huấn luyện trên **15 nghìn
> câu** NQ-train. Có thêm biến thể hybrid: đoạn top-1 của reranker giữ dạng văn bản.
>
> Quan trọng: nhóm đặt **cổng go/no-go trước khi chạy** – trên 200 câu dev, phải hơn closed-book
> ít nhất 10 điểm và đạt 70% accuracy của văn bản đầy đủ; không đạt thì dừng và báo cáo phân
> tích. Để so sánh: PISCO huấn luyện trên hơn 450 nghìn mẫu với model 7B.

*(Chuyển người nói: "Tiếp theo, bạn B sẽ trình bày kết quả.")*

## Slide 8 – Kết quả 1: giữ bằng chứng (1:15)

> Trước khi gọi LLM, nhóm đo xem mỗi bộ nén giữ lại đáp án trong ngữ cảnh bao nhiêu. Ở cùng mức
> khoảng 4 lần, **Provence giữ 91,6%** bằng chứng, hơn rerank top-5 khoảng 12 điểm và hơn
> LLMLingua-2 khoảng 24 điểm. BM25 chỉ 53% vì 19 đoạn nhiễu trong bộ này là hard negative – trùng
> từ khoá với câu hỏi.
>
> Ý chính: context recall là **cận trên** của accuracy sau nén, và đo được miễn phí trên CPU – nhờ
> vậy nhóm loại sớm các cấu hình không đáng chạy.

## Slide 9 – Kết quả 2: accuracy theo số token (1:30)

> *[chỉ ngôi sao góc phải trên]* Ngữ cảnh đầy đủ đạt **65,3%**, closed-book 38,8%, oracle 92,6%.
>
> *[chỉ góc trái trên]* Hai điểm tốt nhất nằm ở đây: **Provence** nén 4,6 lần đạt **64,0%**,
> **LongLLMLingua** 4,3 lần đạt **64,4%** – cả hai đều không giảm có ý nghĩa thống kê so với
> ngữ cảnh đầy đủ, nhưng Provence nén nhanh hơn khoảng 3 lần. Rerank top-5 đạt 62,7%.
>
> *[chỉ góc trái dưới]* Ngược lại, **LLMLingua-2 ở 4 lần chỉ còn 38,3% – bằng closed-book**, dù
> recall vẫn còn 68%. Phân tích lỗi cho thấy văn bản sau khi xoá token bị vụn, reader 7B không đọc
> được dù đáp án vẫn ở đó.
>
> Ý chính: **nén theo câu hỏi** giảm 4 lần token gần như không mất độ chính xác; nén mức token làm
> hỏng văn bản.

## Slide 10 – Kết quả 3: lost in the middle (1:15)

> Đây là câu hỏi thú vị nhất. Với Qwen2.5-7B, ngữ cảnh đầy đủ đạt 65,6 / 62,2 / 68,0 ở ba vị trí –
> gần như **phẳng**, chênh 5,8 điểm. Còn GPT-3.5 trong bài báo là chữ U rõ: chênh **21,6 điểm**.
>
> Provence và LongLLMLingua chạy **song song** với đường Full – không còn nhiều để "làm phẳng".
>
> *[chỉ đường xám đứt]* Cắt cụt cho kết quả đặc biệt: ở vị trí 1 đạt 74% – cao hơn full 8,4 điểm vì
> bớt nhiễu – nhưng ở vị trí 10, 20 rớt xuống 19,6% vì đáp án bị cắt mất.
>
> Ý chính: reader 7B hiện đại hầu như không còn lost in the middle ở độ dài 3 nghìn token.

## Slide 11 – Kết quả 4: so với số liệu công bố (1:00)

> Bảng so độ chênh so với baseline của chính mỗi hệ, không so số tuyệt đối, vì GPT-3.5-0613 đã
> ngừng.
>
> **Tái lập được**: BM25 kém xa dense; LLMLingua-2 đơn lẻ ở 4 lần sụt nặng – nhóm đo −21 đến −30
> điểm, bài báo −11 đến −27.
>
> **Không tái lập được**: kết luận "nén làm tăng accuracy ở giữa". LongLLMLingua công bố **+17,1
> điểm** ở vị trí 10, nhóm chỉ đo được **+1,0, không có ý nghĩa**. Lý do chính là slide trước: reader
> 7B mới không còn lost in the middle – và như sẽ thấy, reader 1.5B thì vẫn còn.

## Slide 12 – Kết quả 5a: PISCO (1:00)

> Với nén mềm, trước hết nhóm thử checkpoint công bố PISCO trên Mistral-7B. 20 đoạn thành 160
> vector – nén **18 lần** – đạt 48,2% so với 60,3% văn bản: giảm 9 đến 17 điểm, tệ nhất khi đoạn
> vàng ở cuối.
>
> Nếu rerank trước rồi chỉ nén 5 đoạn: 58,5% với 40 vector, tức **72 lần**, ngang văn bản ở vị trí 1
> nhưng kém 6–7 điểm ở vị trí 10, 20. Vẫn cao hơn closed-book 12 điểm – vector có mang thông tin,
> nhưng chưa đủ thay văn bản.

## Slide 13 – Kết quả 5b: mô hình của nhóm (1:00)

> Mô hình encode-once của nhóm: loss giảm từ 7,4 xuống 1,2, nhưng **không qua cổng go/no-go** – dev
> 22% so với closed-book 19%. Nhóm dừng theo đúng quy tắc đặt trước, chỉ đo thêm để phân tích.
>
> *[chỉ bảng chẩn đoán]* Chỉ đưa đoạn vàng dạng vector: 23%; dạng văn bản: 72%; văn bản vàng cộng 19
> đoạn vector: 70%. Vậy **nút thắt là 8 vector không giữ nổi đáp án của một đoạn**, không phải do
> nhiễu. Biến thể hybrid 48,5% gần bằng chỉ đọc top-1 văn bản – lợi ích đến từ đoạn được chọn.
>
> Kết quả phụ quan trọng: với reader 1.5B nhạy vị trí, **Provence tăng 7,5 điểm** so với ngữ cảnh
> đầy đủ – tức "nén làm tăng accuracy" vẫn đúng với reader nhỏ.

## Slide 14 – Kết quả 6: truy hồi hiệu quả (1:00)

> Track B: kho gộp **50,7 nghìn đoạn**, 500 câu. BM25 và embedding tĩnh đạt khoảng 55% hit@5.
> bge-small đạt **76,2%**. Điểm đáng chú ý là lượng tử hoá: **int8** giảm chỉ mục từ 74 MB xuống
> **18,6 MB mà không mất điểm nào**; binary kèm rescore còn **2,3 MB** – nhỏ hơn 32 lần – chỉ mất
> 2,8 điểm.

## Slide 15 – Kết quả 7: end-to-end (0:45)

> Ghép cả hai track: bge-small int8 lấy top-20 từ kho, rồi nén, rồi Qwen-7B trả lời. Top-20 chứa
> đáp án ở 95,4% câu, accuracy **66,0%**, ngang thiết lập 20 đoạn chuẩn. Ở mức nén 4 lần,
> **LongLLMLingua 65,6%** gần như không mất gì, rerank top-5 và Provence mất 3 đến 3,6 điểm – đều
> không có ý nghĩa; Dense top-5 mất 9 điểm, LLMLingua-2 mất gần 30 điểm. Thứ tự giữa các bộ nén
> về cơ bản giống thí nghiệm trên 20 đoạn chuẩn.
>
> Tức là: chỉ mục 18,6 MB cộng bộ nén chạy CPU giữ gần nguyên chất lượng ở **một phần tư số token**.

## Slide 16 – Kết quả 8: không có GPU (0:45)

> Đo độ trễ nén trên RTX 4070 SUPER và CPU i5 8 luồng. Mức tài liệu dưới **0,4 giây** trên CPU –
> dùng tương tác được. Mức câu chậm hơn 21–25 lần: Provence **4,9 giây**. Mức token 13–15 giây:
> LongLLMLingua ngang Provence về accuracy nhưng chậm gấp 3.
>
> Kết luận thực tế: không có GPU vẫn làm được – nén trên CPU, reader gọi qua API; Provence giảm
> **78% token đầu vào** mà chỉ mất 1,3 điểm, không có ý nghĩa.

*(Chuyển người nói: "Sau đây là demo và kết luận.")*

## Slide 17 – Demo (0:45, nếu demo trực tiếp: 2:00)

> Nhóm có demo Gradio hai tab. *[chỉ ảnh trái]* Tab đầu: chọn một câu hỏi, chọn vị trí đoạn vàng và
> các bộ nén để so sánh cạnh nhau. *[chỉ ảnh phải]* Mỗi bộ nén hiện số token trước/sau, thời gian
> nén, đáp án còn trong ngữ cảnh không; câu được giữ tô xanh, câu bị loại gạch ngang. Tab thứ hai
> nhận câu hỏi tự do, truy hồi trên kho 50,7 nghìn đoạn rồi nén và trả lời. Toàn bộ chạy trên CPU.

*(Demo trực tiếp chỉ khi còn ≥ 3 phút: mở sẵn `uv run python app/demo.py` trước buổi, chọn một câu
hỏi có đoạn vàng ở vị trí 10, so Provence với LLMLingua-2 r=0,25 để thấy văn bản vụn.)*

## Slide 18 – Hạn chế & kết luận (1:00)

> Về **hạn chế**: nhóm dùng một LLM 7B lượng tử hoá và một tập dữ liệu NQ; 500 câu, một seed; chưa
> đo ngữ cảnh rất dài; nén mềm chỉ huấn luyện ở quy mô nhỏ.
>
> Bốn **kết luận**:
> 1. **Nén theo câu hỏi tốt hơn chọn tài liệu, tốt hơn xoá token**: Provence và LongLLMLingua nén
>    hơn 4 lần, mất không quá 1,3 điểm; LLMLingua-2 về mức closed-book.
> 2. Reader 7B gần như hết lost in the middle, nên "nén làm tăng accuracy" chỉ còn đúng với reader
>    nhỏ.
> 3. Không có GPU vẫn làm được: nén trên CPU, reader qua API, dense nhỏ int8 cho truy hồi.
> 4. Nén mềm chưa thay được văn bản: PISCO mất 9–17 điểm, mô hình 1.5B của nhóm không qua cổng.
>
> Chúng em xin cảm ơn thầy cô và các bạn đã lắng nghe, nhóm sẵn sàng nhận câu hỏi.

---

## Chuẩn bị hỏi đáp

| Câu hỏi có thể gặp | Ý trả lời |
|---|---|
| Vì sao không dùng GPT-3.5 như bài báo? | Bản 0613 đã ngừng; nhóm dùng một LLM mở cố định cho mọi phương pháp và chỉ so **độ chênh** so với baseline của chính mình. |
| LLMLingua-2 còn 68% recall sao accuracy bằng closed-book? | Phân tích lỗi: 19,0% câu hỏng dù đáp án vẫn còn trong ngữ cảnh – văn bản sau khi xoá token bị vụn, reader 7B không dùng được (`tables/errors.tex`). |
| Provence và LongLLMLingua ngang nhau, chọn cái nào? | Provence: nén nhanh hơn ~3 lần (200 ms so với 625 ms trên GPU; 4,9 s so với 14,8 s trên CPU), tỉ lệ nén cao hơn (4,6× so với 4,3×). |
| "Không có ý nghĩa" nghĩa là gì? | Paired bootstrap trên cùng 500 câu, khoảng tin cậy 95% của độ chênh chứa 0. |
| Sao nén mềm của nhóm thất bại? | Quy mô: 15k mẫu, 1.5B so với >450k mẫu, 7B của PISCO. Chẩn đoán cho thấy 8 vector không giữ nổi đáp án của một đoạn (23% so với 72% văn bản). |
| Sao không thử nhiều seed/tập dữ liệu khác? | Giới hạn GPU (một card 12 GB, một job nặng mỗi lúc); nhóm ưu tiên đủ 3 vị trí × 500 câu để so được với số công bố. |
| Kết quả có áp dụng cho ngữ cảnh dài hơn? | Chưa đo (đã nêu trong hạn chế); ở 3K token, reader 7B gần như không nhạy vị trí. |
