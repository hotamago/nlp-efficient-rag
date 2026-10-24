"""P0: evaluate PISCO (Louis et al., arXiv 2501.16075) on the LITM NQ benchmark.

PISCO = Mistral-7B-Instruct-v0.2 + encoder/decoder LoRA adapters; each document (<=128 tokens) is
compressed once into 8 memory vectors (16x). Rows written to results/compression with
llm_model "mistral-7b-nf4+pisco" (compressed) or "mistral-7b-nf4" (same backbone, adapters off,
documents as text, same PISCO prompt) so the compression effect is measured on one backbone.
Note: the text rows still use PISCO's retrained first/last layers (embeddings, lm_head), i.e.
they are a same-backbone control, not the stock Mistral-7B-Instruct-v0.2.

    uv run python scripts/eval_pisco.py --n 500 --positions 0 9 19
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

from effrag.compression.base import render_docs
from effrag.config import DATA_DIR, SEED, limit_gpu_memory
from effrag.data import Doc, Example, load_litm_nq
from effrag.metrics import best_subspan_em, exact_match, f1
from effrag.results import new_run_id, record_run
from effrag.tokens import count_tokens

REPO = "naver/pisco-mistral"
REVISION = "2121d0c0e8aa67328930e9853a217618e3725b7b"
SYSTEM = (
    "You are a helpful assistant. Your task is to extract relevant information from provided "
    "documents and to answer to questions as briefly as possible."
)
DOC_MAX_TOKENS = 128
N_MEM = 8  # doc_max_length 128 / compr_rate 16
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_pisco():
    """Local copy of the snapshot with quantization=int4 (nf4) so the 7B fits next to nothing."""
    from transformers import AutoModel

    src = Path(snapshot_download(REPO, revision=REVISION))
    dst = DATA_DIR / "pisco_nf4"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != "config.json" and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization"] = "int4"
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    model = AutoModel.from_pretrained(
        str(dst), trust_remote_code=True, attn_implementation="sdpa", device_map={"": 0}
    )
    model.eval()
    return model


def text_prompt(tok, question: str, docs: list[Doc]) -> str:
    """PISCO prompt with documents as text (adapters off): the uncompressed reference."""
    background = "\n".join(d.render() for d in docs)
    user = f"Background:\n{background}\n\nQuestion:{question}" if docs else f"Question:{question}"
    # Mistral's template has no system role -> merged into the user turn, as PISCO does
    messages = [{"role": "user", "content": SYSTEM + "\n" + user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_text(model, prompts: list[str], max_new_tokens: int) -> list[str]:
    tok = model.decoder_tokenizer
    inp = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    model.decoder.disable_adapters()
    try:
        out = model.decoder.generate(
            **inp, do_sample=False, top_p=None, temperature=None, max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )  # fmt: skip
    finally:
        model.decoder.enable_adapters()
        model.set_all_adapters()
    return tok.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)


def encoder_view(tok, text: str) -> str:
    """The part of a document PISCO compresses: its first doc_max_length (128) tokens."""
    ids = tok(text, add_special_tokens=False)["input_ids"][:DOC_MAX_TOKENS]
    return tok.decode(ids)


def select(examples: list[Example], variant: str, reranker) -> list[list[Doc]]:
    if variant in ("all", "full"):
        return [ex.docs for ex in examples]
    if variant == "closedbook":
        return [[] for _ in examples]
    if variant == "oracle":
        return [[d for d in ex.docs if d.is_gold][:1] for ex in examples]
    k = int(variant.removeprefix("rerank"))
    out = []
    for ex in examples:
        s = reranker.predict([(ex.question, d.render()) for d in ex.docs])
        top = set(np.argsort(-s)[:k].tolist())
        out.append([d for i, d in enumerate(ex.docs) if i in top])  # keep original order
    return out


def run(model, mode: str, variant: str, pos: int, n: int, bs: int, reranker, run_id: str):
    examples = load_litm_nq(pos, n=n, seed=SEED)
    selected = select(examples, variant, reranker)
    rows = []
    for b in tqdm(range(0, len(examples), bs), desc=f"{mode}/{variant} gold@{pos}"):
        exs, sel = examples[b : b + bs], selected[b : b + bs]
        t0 = time.perf_counter()
        if mode == "pisco":
            preds = model.generate_from_text(
                [ex.question for ex in exs], [[d.render() for d in s] for s in sel],
                max_new_tokens=64,
            )  # fmt: skip
        else:
            prompts = [
                text_prompt(model.decoder_tokenizer, ex.question, s)
                for ex, s in zip(exs, sel, strict=True)
            ]  # noqa: E501
            preds = generate_text(model, prompts, max_new_tokens=64)
        dt = (time.perf_counter() - t0) / len(exs)
        if b == 0:
            print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        for ex, s, pred in zip(exs, sel, preds, strict=True):
            kept = render_docs(s)
            if mode == "pisco":  # recall over what the encoder actually sees (<=128 tokens/doc)
                kept = "\n".join(encoder_view(model.decoder_tokenizer, d.render()) for d in s)
            comp = N_MEM * len(s) if mode == "pisco" else count_tokens(kept)
            rows.append({
                "run_id": run_id, "qid": ex.qid, "gold_pos": pos, "question": ex.question,
                "answers": ex.answers, "orig_tokens": count_tokens(render_docs(ex.docs)),
                "comp_tokens": comp, "ctx_recall": best_subspan_em(kept, ex.answers),
                "compress_s": 0.0, "llm_s": dt, "prediction": pred.strip(),
                "accuracy": best_subspan_em(pred, ex.answers),
                "em": exact_match(pred, ex.answers), "f1": f1(pred, ex.answers),
            })  # fmt: skip
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--positions", type=int, nargs="+", default=[0, 9, 19])
    ap.add_argument("--bs", type=int, default=1, help="batch size for PISCO (compressed) configs")
    ap.add_argument("--bs-text", type=int, default=1, help="batch size for full-text configs")
    ap.add_argument("--track", default="compression", help="e.g. compression_smoke for tests")
    ap.add_argument(
        "--configs", nargs="+",
        default=["pisco:all", "text:full", "text:closedbook", "text:oracle", "pisco:rerank5",
                 "text:rerank5"],
        help="mode:variant[@pos,...]; @ overrides --positions for that config",
    )  # fmt: skip
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder

    limit_gpu_memory()
    reranker = CrossEncoder(RERANKER, device="cuda")
    model = load_pisco()
    for cfg in args.configs:
        cfg, _, at = cfg.partition("@")
        mode, variant = cfg.split(":")
        method = "pisco" if mode == "pisco" else "mistral_text"
        llm = "mistral-7b-nf4+pisco" if mode == "pisco" else "mistral-7b-nf4"
        params = {"docs": variant} | ({"rerank": RERANKER} if variant.startswith("rerank") else {})
        run_id = new_run_id(f"{method}_{variant}")
        positions = [int(x) for x in at.split(",")] if at else args.positions
        if variant in ("closedbook", "oracle"):
            positions = [0]
        for pos in positions:
            bs = args.bs if mode == "pisco" else args.bs_text
            try:  # PISCO's generate_from_text does not disable autograd itself
                with torch.inference_mode():
                    rows = run(model, mode, variant, pos, args.n, bs, reranker, run_id)
            except torch.OutOfMemoryError as e:  # skip this config, keep the loaded model
                print(f"OOM in {cfg}@{pos}: {e}", flush=True)
                torch.cuda.empty_cache()
                continue
            record_run(rows, run_id=run_id, method=method, params=params, gold_pos=pos,
                       llm_model=llm, seed=SEED, track=args.track)  # fmt: skip


if __name__ == "__main__":
    main()
