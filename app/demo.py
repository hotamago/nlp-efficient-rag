"""Gradio demo: compare context compressors side by side and run retrieve -> compress -> answer.

uv run python app/demo.py            # http://localhost:7860
"""

import html
import os
import re
import time
from functools import cache

import gradio as gr

from effrag.compression import DocwiseCompressor, get_compressor
from effrag.config import LLMSettings
from effrag.data import Doc, litm_corpus, load_litm_nq
from effrag.generation import build_prompt
from effrag.metrics import best_subspan_em
from effrag.tokens import count_tokens

OP = "hotchpotch/open-provence-reranker-v1-gte-modernbert-base"
METHODS = {
    "Full (không nén)": ("full", {}),
    "Rerank top-5 (MiniLM)": ("rerank_topk", {"k": 5}),
    "LLMLingua-2 (r=0.25)": ("llmlingua2", {"rate": 0.25}),
    "LongLLMLingua (730 tok)": (
        "longllmlingua",
        {"target_token": 730, "device": os.getenv("EFFRAG_DEMO_LONGLLMLINGUA_DEVICE")},
    ),
    "Provence (τ=0.1)": ("provence", {"threshold": 0.1}),
    "OpenProvence (τ=0.1)": ("provence", {"threshold": 0.1, "model": OP}),
}
DEFAULT_METHODS = ["Full (không nén)", "LLMLingua-2 (r=0.25)", "Provence (τ=0.1)"]


@cache
def compressor(label: str):
    name, params = METHODS[label]
    return get_compressor(name, **params)


@cache
def examples(gold_pos: int):
    return load_litm_nq(gold_pos, n=200)


@cache
def llm():
    from effrag.generation import LLM

    try:
        client = LLM(LLMSettings())
        client.client.models.list()
        return client
    except Exception as e:  # server down / no key: demo still shows compression
        print(f"LLM unavailable: {e}")
        return None


@cache
def retriever(kind: str):
    from effrag.retrieval import BM25Retriever, DenseRetriever

    corpus = litm_corpus(0)
    if kind == "BM25S":
        return corpus, BM25Retriever(corpus)
    return corpus, DenseRetriever(
        corpus,
        model="BAAI/bge-small-en-v1.5",
        query_prompt="Represent this sentence for searching relevant passages: ",
    )


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def render_docwise(docs: list[Doc], kept: list[str], answers: list[str]) -> str:
    """Original passages; kept sentences highlighted, dropped ones greyed out."""
    out = []
    for i, (d, k) in enumerate(zip(docs, kept, strict=True), 1):
        k_norm = _norm(k)
        parts = []
        for s in _sentences(d.text):
            esc = html.escape(s)
            if k and _norm(s) and _norm(s) in k_norm:
                parts.append(f"<mark style='background:#c8f7c5'>{esc}</mark>")
            else:
                parts.append(
                    f"<span style='color:#9a9a9a;text-decoration:line-through'>{esc}</span>"
                )
        badge = " <b style='color:#a31621'>[đoạn vàng]</b>" if d.is_gold else ""
        status = "giữ" if k else "<span style='color:#c0392b'>bị loại</span>"
        out.append(
            f"<div style='margin:4px 0'><b>[{i}] {html.escape(d.title)}</b> ({status}){badge}<br>"
            + " ".join(parts) + "</div>"
        )  # fmt: skip
    return "".join(out)


def run_one(label: str, question: str, docs: list[Doc], answers: list[str]) -> tuple[str, str]:
    comp = compressor(label)
    t0 = time.perf_counter()
    if isinstance(comp, DocwiseCompressor):
        comp._memo.clear()
        kept, _ = comp.compress_docs(question, docs)
        context = "\n".join(f"Document [{i}]{s}" for i, s in enumerate([k for k in kept if k], 1))
        body = render_docwise(docs, kept, answers)
    else:
        context = comp.compress(question, docs).context
        body = f"<pre style='white-space:pre-wrap'>{html.escape(context)}</pre>"
    comp_s = time.perf_counter() - t0
    orig, new = count_tokens("\n".join(d.render() for d in docs)), count_tokens(context)
    answer, ok = "(LLM không khả dụng)", ""
    client = llm()
    if client is not None:
        out = client.generate(build_prompt(question, context))
        answer = out["text"].strip()
        if answers:
            ok = " (đúng)" if best_subspan_em(answer, answers) else " (sai)"
    header = (
        f"### {label}\n**Token:** {orig} → {new} (nén {orig / max(new, 1):.1f}×) · "
        f"**Thời gian nén:** {1000 * comp_s:.0f} ms · **Còn đáp án trong ngữ cảnh:** "
        f"{'có' if best_subspan_em(context, answers) else 'không'}\n\n"
        f"**Trả lời:** {answer}{ok}"
    )
    return header, body


def compare(idx: int, gold_pos: int, labels: list[str]):
    ex = examples(int(gold_pos))[int(idx)]
    info = f"**Câu hỏi:** {ex.question}  \n**Đáp án chuẩn:** {', '.join(ex.answers)}"
    blocks = []
    for label in labels:
        header, body = run_one(label, ex.question, ex.docs, ex.answers)
        blocks.append((header, body))
    md = "\n\n---\n\n".join(h for h, _ in blocks)
    bodies = "".join(
        f"<details><summary><b>{html.escape(lbl)}</b> – ngữ cảnh sau nén</summary>{b}</details>"
        for lbl, (_, b) in zip(labels, blocks, strict=True)
    )
    return info, md, bodies


def rag(question: str, kind: str, k: int, label: str):
    corpus, r = retriever(kind)
    t0 = time.perf_counter()
    ids = r.search([question], k=int(k))[0]
    ret_ms = 1000 * (time.perf_counter() - t0)
    docs = [corpus[i] for i in ids]
    header, body = run_one(label, question, docs, [])
    return (
        f"**Truy hồi {kind}:** top-{int(k)} trong {len(corpus):,} đoạn, {ret_ms:.1f} ms\n\n"
        + header,
        body,
    )


with gr.Blocks(title="Efficient RAG demo") as demo:
    gr.Markdown(
        "# Efficient RAG: nén ngữ cảnh & truy hồi hiệu quả\n"
        "BTL IT4772 – NQ-Open 20 tài liệu (Lost in the Middle). "
        "Câu được giữ tô xanh, câu bị loại gạch xám."
    )
    with gr.Tab("So sánh bộ nén"):
        with gr.Row():
            idx = gr.Slider(0, 199, value=3, step=1, label="Câu hỏi #")
            pos = gr.Dropdown([0, 4, 9, 14, 19], value=9, label="Vị trí đoạn vàng (0 = đầu)")
        labels = gr.CheckboxGroup(list(METHODS), value=DEFAULT_METHODS, label="Bộ nén")
        btn = gr.Button("Chạy", variant="primary")
        info = gr.Markdown()
        summary = gr.Markdown()
        details = gr.HTML()
        btn.click(compare, [idx, pos, labels], [info, summary, details])
    with gr.Tab("RAG end-to-end"):
        q = gr.Textbox(
            label="Câu hỏi (tiếng Anh)",
            value="when was the nobel prize for literature first awarded",
        )
        with gr.Row():
            kind = gr.Radio(["BM25S", "Dense (bge-small)"], value="BM25S", label="Retriever")
            k = gr.Slider(5, 20, value=10, step=1, label="Top-k")
            lab = gr.Dropdown(list(METHODS), value="Provence (τ=0.1)", label="Bộ nén")
        btn2 = gr.Button("Hỏi", variant="primary")
        out_md = gr.Markdown()
        out_html = gr.HTML()
        btn2.click(rag, [q, kind, k, lab], [out_md, out_html])

if __name__ == "__main__":
    demo.launch()
