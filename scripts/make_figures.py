"""Build report tables (LaTeX) and figures (PDF/PNG) from results/.

uv run python scripts/make_figures.py
"""

import json
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from effrag.config import RESULTS_DIR, ROOT

FIG = ROOT / "docs" / "report" / "figures"
TAB = ROOT / "docs" / "report" / "tables"
SLIDE_FIG = ROOT / "docs" / "slide" / "figures"
POS = [0, 9, 19]
POS_LABEL = ["1", "10", "20"]
POS_INVARIANT = {"closedbook", "oracle"}  # run at one position only
POS_LABELS_INV = {"Closed-book", "Oracle"}

# Reference palette (dataviz skill), light mode, fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY, INK, INK2 = "#8a8985", "#0b0b0b", "#52514e"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c9c8c3",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#ebeae6", "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200,
})  # fmt: skip

BASE = "Full (20 docs)"
MAIN_LLM = "qwen2.5-7b-instruct"  # reader of the main table; other readers go to soft_table()
OP = "hotchpotch/open-provence-reranker-v1-gte-modernbert-base"


def label(method: str, params: str) -> str:
    p = json.loads(params)
    if method == "truncate":
        return f"Truncate {p['budget']}"
    if method.endswith("_topk"):
        base = {"bm25_topk": "BM25", "dense_topk": "Dense", "rerank_topk": "Rerank"}[method]
        return f"{base} top-{p['k']}"
    if method == "llmlingua2":
        return f"LLMLingua-2 r={p['rate']}"
    if method == "longllmlingua":
        return f"LongLLMLingua {p['target_token']}t"
    if method == "provence":
        name = "OpenProvence" if p.get("model") == OP else "Provence"
        return f"{name} τ={p['threshold']}"
    if method == "chain":
        return "Rerank-10 → LLMLingua-2"
    if method in ("pisco", "mistral_text"):
        docs = {"all": "20 đoạn", "full": "20 đoạn", "rerank5": "rerank top-5"}.get(p["docs"])
        if docs is None:
            return {"closedbook": "Closed-book", "oracle": "Oracle"}[p["docs"]]
        return f"PISCO 16× ({docs})" if method == "pisco" else f"Văn bản ({docs})"
    if method == "soft":
        return f"Nén mềm ({'lai top-1' if p['variant'] == 'hybrid' else '20 đoạn'})"
    return {"full": "Full (20 docs)", "closedbook": "Closed-book", "oracle": "Oracle"}[method]


FAMILY = {
    "full": "Tham chiếu", "closedbook": "Tham chiếu", "oracle": "Tham chiếu",
    "truncate": "Prompt", "bm25_topk": "Tài liệu", "dense_topk": "Tài liệu",
    "rerank_topk": "Tài liệu", "llmlingua2": "Token", "longllmlingua": "Token",
    "chain": "Token", "provence": "Câu",
}  # fmt: skip


def load(track: str, n: int | None = 500, llm: str | None = MAIN_LLM) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / track / "summary.csv")
    if n is not None:
        df = df[df.n == n]
    if llm is not None and track == "compression":
        df = df[df.llm_model == llm]
    # keep the latest run per (reader, method, params, gold_pos)
    key = ["llm_model", "method", "params", "gold_pos"]
    df = df.sort_values("date").drop_duplicates(key, keep="last")
    df["label"] = [label(m, p) for m, p in zip(df.method, df.params, strict=True)]
    df["family"] = df.method.map(FAMILY)
    # position-invariant references are run once and shown at every position
    extra = [
        df[(df.label == lab) & (df.llm_model == llm)].assign(gold_pos=pos)
        for lab, llm in df[df.label.isin(POS_LABELS_INV)][["label", "llm_model"]]
        .drop_duplicates()
        .itertuples(index=False)
        for pos in POS
        if not ((df.label == lab) & (df.llm_model == llm) & (df.gold_pos == pos)).any()
    ]
    return pd.concat([df, *extra], ignore_index=True) if extra else df


def pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    t = df.pivot_table(index="label", columns="gold_pos", values=value)
    return t.reindex(columns=POS)


def save(fig, name: str) -> None:
    for d in (FIG, SLIDE_FIG):
        d.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(SLIDE_FIG / f"{name}.png")
    plt.close(fig)


RETRIEVER_NAME = {
    "bm25": "BM25S", "static-1024": "Static 1024d", "static-256": "Static 256d (MRL)",
    "bge-small": "bge-small (flat)", "bge-small-hnsw": "bge-small HNSW",
    "bge-small-int8": "bge-small int8", "bge-small-binary": "bge-small binary + rescore",
}  # fmt: skip


def vi_num(text: str) -> str:
    """Vietnamese number style (decimal comma, dot thousands); keeps names like Qwen2.5-1.5B."""
    text = re.sub(r"(?<![\w.])(\d+)\.(\d+)(?![\w.])", r"\1,\2", text)
    return re.sub(r"(?<![\w,.])(\d{1,3})(\d{3})(?![\w,.])", r"\1.\2", text)


def disp(name: str) -> str:
    """Display label of a run; internal labels (keys in results/) stay unchanged."""
    alias = {"Văn bản (20 đoạn)": "Full (20 đoạn)", "Văn bản (rerank top-5)": "Rerank top-5"}
    return alias.get(name, name.replace("20 docs", "20 đoạn").replace("lai top-1", "hybrid top-1"))


def fig_label(name: str) -> str:
    return vi_num(disp(name))


def tex_name(name: str) -> str:
    """Table label; numbers are localised once, by write_tex()."""
    return disp(name).replace("τ", "$\\tau$").replace("→", "$\\rightarrow$")


def write_tex(name: str, lines: list[str]) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text(vi_num("\n".join(lines)) + "\n")


def fmt(v: float, nd: int = 1) -> str:
    return "--" if pd.isna(v) else f"{v:.{nd}f}"


def main_table(df: pd.DataFrame) -> pd.DataFrame:
    acc = pivot(df, "accuracy") * 100
    t = acc.copy()
    t.columns = POS_LABEL
    t["TB"] = acc.mean(axis=1, skipna=False)  # only when all positions were run
    g = df.groupby("label")
    t["Token"] = g.comp_tokens.mean()
    t["Nén"] = g.ratio.mean()
    t["Recall"] = g.ctx_recall.mean() * 100
    t["ms"] = g.compress_ms.mean()
    t["family"] = g.family.first()
    return t


def write_latex(t: pd.DataFrame) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    order = ["Tham chiếu", "Prompt", "Tài liệu", "Token", "Câu"]
    best_tb = t[t.family != "Tham chiếu"]["TB"].max()
    lines = []
    for fam in order:
        sub = t[t.family == fam].sort_values("Token", ascending=False)
        if sub.empty:
            continue
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{fam}}}}} \\\\")
        for name, r in sub.iterrows():
            accs = " & ".join(fmt(v) for v in r[POS_LABEL])
            ratio = "--" if r.Token < 1 else f"{r['Nén']:.1f}$\\times$"
            tb = fmt(r.TB)
            if fam != "Tham chiếu" and abs(r.TB - best_tb) < 1e-9:
                tb = f"\\best{{{tb}}}"
            lines.append(
                f"\\quad {tex_name(name)} & {accs} & {tb} & {r.Token:.0f} & "
                f"{ratio} & {r.Recall:.1f} & {fmt(r.ms, 0)} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    write_tex("main_results.tex", lines)


def recall_table() -> pd.DataFrame:
    """LLM-free view of every compressor run (dry runs): tokens, ratio, recall, latency."""
    df = load("compression_dryrun", n=None)
    df = df[df.gold_pos.isin(POS) & ~df.method.isin(POS_INVARIANT | {"full"})]
    g = df.groupby(["label", "n"])
    t = g.agg(family=("family", "first"), Token=("comp_tokens", "mean"),
              ratio=("ratio", "mean"), ms=("compress_ms", "mean"))  # fmt: skip
    rec = df.pivot_table(index=["label", "n"], columns="gold_pos", values="ctx_recall") * 100
    t = t.join(rec.reindex(columns=POS)).reset_index()
    order = {"Prompt": 0, "Tài liệu": 1, "Token": 2, "Câu": 3}
    t = t.sort_values(["family", "Token"], key=lambda s: s.map(order) if s.name == "family" else -s)
    lines = []
    for _, r in t.iterrows():
        recalls = " & ".join(fmt(r[p]) for p in POS)
        note = "" if r.n == 500 else f" % n={r.n}"
        lines.append(
            f"{tex_name(r.label)} & {r.Token:.0f} & {r.ratio:.1f}$\\times$ & {recalls} & "
            f"{fmt(r.ms, 0)} \\\\{note}"
        )
    TAB.mkdir(parents=True, exist_ok=True)
    write_tex("recall.tex", lines)
    return t


# Published GPT-3.5-Turbo numbers on the same benchmark (positions 1/10/20 of 20),
# from LongLLMLingua (Jiang et al., 2024, Tab. 1) and LLMLingua-2 (Pan et al., 2024, Tab. 4).
PUBLISHED = [
    ("Full (20 docs)", "Original prompt", [75.7, 54.1, 63.1]),
    ("BM25 top-5", "BM25 (4$\\times$)", [40.6, 38.2, 36.6]),
    ("Dense top-5", "SBERT (4$\\times$)", [66.9, 59.0, 60.3]),
    ("LLMLingua-2 r=0.25", "LLMLingua-2 (4$\\times$)", [48.6, 43.6, 39.9]),
    ("Rerank-10 → LLMLingua-2", "LLMLingua-2 + lọc tài liệu (4$\\times$)", [74.0, 67.0, 65.3]),
    ("LongLLMLingua 730t", "LongLLMLingua (4$\\times$)", [75.0, 71.2, 74.7]),
]


def published_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy change vs. each side's own uncompressed baseline (published GPT-3.5 vs. ours).

    Configs run on a different sample are compared on the questions they share with the
    full-context run (paired), not against the n=500 mean.
    """
    ours = pivot(df, "accuracy") * 100
    base_pub = PUBLISHED[0][2]
    lines, out = [], []
    for key, pub_name, pub in PUBLISHED:
        if key == BASE:
            row = [ours.loc[key, p] for p in POS]
            pub_s = " & ".join(f"{v:.1f}" for v in pub)
            our_s = " & ".join(fmt(v) for v in row)
        else:
            row = []
            for p in POS:
                ref, q = per_question(BASE, p), per_question(key, p, n=None)
                if ref is None or q is None:
                    row.append(float("nan"))
                    continue
                common = ref.index.intersection(q.index)
                row.append(100 * (q.loc[common, "accuracy"].mean()
                                  - ref.loc[common, "accuracy"].mean()))  # fmt: skip
            pub_s = " & ".join(f"{v - b:+.1f}" for v, b in zip(pub, base_pub, strict=True))
            our_s = " & ".join("--" if pd.isna(v) else f"{v:+.1f}" for v in row)
        lines.append(f"{pub_name} & {pub_s} & {tex_name(key)} & {our_s} \\\\")
        out.append({"published": pub_name, "ours": key, **dict(zip(POS_LABEL, row, strict=True))})
    write_tex("published.tex", lines)
    return pd.DataFrame(out)


def per_question(method_label: str, pos: int, n: int | None = 500) -> pd.DataFrame | None:
    """Per-question rows of the latest LLM run for a config label at a gold position."""
    for path in sorted((RESULTS_DIR / "compression").glob(f"*_gold{pos}.jsonl"), reverse=True):
        rows = [json.loads(x) for x in path.open()]
        r0 = rows[0]
        run = r0["run_id"]
        summ = pd.read_csv(RESULTS_DIR / "compression" / "summary.csv")
        s = summ[(summ.run_id == run) & (summ.gold_pos == pos) & (summ.llm_model == MAIN_LLM)]
        if n is not None:
            s = s[s.n == n]
        if not s.empty and label(s.method.iloc[0], s.params.iloc[0]) == method_label:
            return pd.DataFrame(rows).set_index("qid")
    return None


def paired_bootstrap(a: pd.Series, b: pd.Series, iters: int = 10000, seed: int = 0):
    """Mean difference (b - a) in points and its 95% CI over paired questions."""
    import numpy as np

    d = (b - a).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(iters, len(d))).mean(axis=1)
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)


def significance(labels: list[str], base: str = BASE) -> pd.DataFrame:
    out = []
    for pos in POS:
        ref = per_question(base, pos)
        for lab in labels:
            q = per_question(lab, pos)
            if ref is None or q is None:
                continue
            common = ref.index.intersection(q.index)
            m, lo, hi = paired_bootstrap(ref.loc[common, "accuracy"], q.loc[common, "accuracy"])
            out.append({"label": lab, "pos": pos + 1, "n": len(common),
                        "delta": m, "lo": lo, "hi": hi})  # fmt: skip
    return pd.DataFrame(out)


SOFT_BLOCKS = [  # (title, readers, file for the per-block copy used by the slides)
    ("Mistral-7B-Instruct-v0.2 (nf4)", ["mistral-7b-nf4", "mistral-7b-nf4+pisco"], "soft_pisco"),
    ("Qwen2.5-1.5B-Instruct", ["qwen2.5-1.5b-instruct", "qwen2.5-1.5b-instruct+soft"], "soft_own"),
]


def soft_table() -> pd.DataFrame | None:
    """Soft compression (PISCO, our encode-once model) next to text baselines of the same reader."""
    df = load("compression", llm=None)
    df = df[df.llm_model != MAIN_LLM]
    if df.empty:
        return None
    lines, out = [], []
    for title, llms, block_file in SOFT_BLOCKS:
        sub = df[df.llm_model.isin(llms)]
        if sub.empty:
            continue
        acc = sub.pivot_table(index="label", columns="gold_pos", values="accuracy") * 100
        acc = acc.reindex(columns=POS)
        g = sub.groupby("label")
        soft = g.llm_model.first().str.contains("+", regex=False)
        t = acc.assign(TB=acc.mean(axis=1, skipna=False), Token=g.comp_tokens.mean(),
                       ratio=g.ratio.mean(), soft=soft)  # fmt: skip
        t = t.sort_values(["soft", "Token"], ascending=[True, False])
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\")
        start = len(lines)
        for name, r in t.iterrows():
            accs = " & ".join(fmt(r[p]) for p in POS)
            ratio = f"{r.ratio:.1f}$\\times$" if r.Token else "--"  # closed-book: no context
            lines.append(f"\\quad {tex_name(name)} & {accs} & {fmt(r.TB)} & {r.Token:.0f} & "
                         f"{ratio} \\\\")  # fmt: skip
        write_tex(f"{block_file}.tex", lines[start:] + ["\\bottomrule"])
        lines.append("\\midrule")
        out.append(t.assign(reader=title))
    lines[-1] = "\\bottomrule"
    TAB.mkdir(parents=True, exist_ok=True)
    write_tex("soft.tex", lines)
    return pd.concat(out)


def fig_position(df: pd.DataFrame) -> None:
    acc = pivot(df, "accuracy") * 100
    series = ["Full (20 docs)", "Provence τ=0.1", "LongLLMLingua 730t", "Rerank top-5",
              "LLMLingua-2 r=0.25", "Rerank-10 → LLMLingua-2"]  # fmt: skip
    series = [s for s in series if s in acc.index]
    if "Truncate 730" in acc.index:
        ax_trunc = acc.loc["Truncate 730"]
    else:
        ax_trunc = None
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for ref, ls in (("Oracle", ":"), ("Closed-book", "--")):
        if ref in acc.index:
            y = acc.loc[ref].mean()
            ax.axhline(y, color=GRAY, ls=ls, lw=1.2)
            ax.text(2.06, y, vi_num(f"{ref} {y:.1f}"), va="center", fontsize=8, color=INK2)
    for i, s in enumerate(series):
        ax.plot(range(len(POS)), acc.loc[s], color=C[i], marker="o", ms=4, label=fig_label(s))
    if ax_trunc is not None:
        ax.plot(range(len(POS)), ax_trunc, color=GRAY, ls="--", marker="s", ms=4, lw=1.4,
                label="Truncate 730")  # fmt: skip
    ax.set_xticks(range(len(POS)), POS_LABEL)
    ax.set_xlabel("Vị trí đoạn chứa đáp án (trên 20 đoạn)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    save(fig, "position_curve")


def fig_tradeoff(t: pd.DataFrame) -> None:
    t = t.dropna(subset=["TB"])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    fams = ["Tài liệu", "Token", "Câu"]
    for i, fam in enumerate(fams):
        sub = t[t.family == fam]
        ax.scatter(sub.Token, sub.TB, s=36, color=C[i], label=f"Mức {fam.lower()}",
                   edgecolor="white", linewidth=1.2, zorder=3)  # fmt: skip
    sub = t[t.family == "Prompt"]
    ax.scatter(sub.Token, sub.TB, s=36, color=GRAY, marker="s", label="Truncate", zorder=3)
    if "Full (20 docs)" in t.index:
        r = t.loc["Full (20 docs)"]
        ax.scatter(r.Token, r.TB, s=70, color=INK, marker="*", label="Full", zorder=3)
    nudge = {"OpenProvence τ=0.1": (4, -9), "Truncate 730": (4, -9), "Rerank top-10": (-30, 6)}
    for name, r in t[t.family != "Tham chiếu"].iterrows():
        ax.annotate(fig_label(name), (r.Token, r.TB), fontsize=6.5, color=INK2,
                    xytext=nudge.get(name, (4, 2)), textcoords="offset points")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("Số token ngữ cảnh trung bình (log)")
    ax.set_ylabel("Accuracy trung bình 3 vị trí (%)")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "tradeoff")


def latency_table() -> pd.DataFrame | None:
    rows = []
    for dev in ("cuda", "cpu"):
        path = RESULTS_DIR / "latency" / f"{dev}.jsonl"
        if path.exists():
            rows += [json.loads(x) for x in path.open()]
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(["device", "method", "params"], keep="last")
    df["label"] = [label(m, p) for m, p in zip(df.method, df.params, strict=True)]
    t = df.pivot_table(index="label", columns="device", values="median_ms")
    t = t.sort_values(t.columns[0])
    lines = []
    for name, r in t.iterrows():
        gpu, cpu = r.get("cuda", float("nan")), r.get("cpu", float("nan"))
        lines.append(f"{tex_name(name)} & {fmt(gpu, 0)} & {fmt(cpu, 0)} & "
                     f"{fmt(cpu / gpu, 1)}$\\times$ \\\\")  # fmt: skip
    write_tex("latency.tex", lines)
    return t


def retrieval_table() -> pd.DataFrame | None:
    path = RESULTS_DIR / "retrieval" / "summary.jsonl"
    if not path.exists():
        return None
    df = pd.DataFrame([json.loads(x) for x in path.open()])
    df = df.drop_duplicates("system", keep="last").set_index("system")
    TAB.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, r in df.iterrows():
        rows.append(
            f"{RETRIEVER_NAME.get(s, s)} & {100 * r['answer_hit@1']:.1f} & "
            f"{100 * r['answer_hit@5']:.1f} & "
            f"{100 * r['answer_hit@20']:.1f} & {100 * r['gold_mrr@10']:.1f} & "
            f"{r.index_mb:.1f} & {r.query_ms:.2f} & {r.query_ms_single:.1f} \\\\"
        )
    write_tex("retrieval.tex", rows)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    names = [RETRIEVER_NAME.get(s, s) for s in df.index]
    ax.barh(names[::-1], 100 * df["answer_hit@5"][::-1], color=C[0], height=0.6)
    for y, v in enumerate(100 * df["answer_hit@5"][::-1]):
        ax.text(v + 0.5, y, vi_num(f"{v:.1f}"), va="center", fontsize=7, color=INK2)
    ax.set_xlabel("Answer-hit@5 (%)")
    ax.grid(axis="y", visible=False)
    save(fig, "retrieval_hit5")
    return df


def fig_pipeline() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9, 2.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis("off")
    boxes = [
        (1, "Câu hỏi", "who got the first\nnobel prize\nin physics?", "#f3f2ee"),
        (18, "Retriever", "(Track B)\nBM25S · bge-small\nstatic · MRL\nint8 / binary · HNSW", "#dce9f8"),  # noqa: E501
        (41, "20 đoạn văn", "~2.900 token\n1 đoạn chứa\nđáp án", "#f3f2ee"),
        (59, "Compressor", "(Track A)\ntop-k · LLMLingua-2\nLongLLMLingua\nProvence · OpenProvence", "#fde3d8"),  # noqa: E501
        (85, "Reader (LLM)", "Qwen2.5-7B\n(vLLM)\n→ câu trả lời", "#d7f1e6"),
    ]  # fmt: skip
    widths = [14, 20, 15, 23, 14]
    for (x, title, body, color), w in zip(boxes, widths, strict=True):
        ax.add_patch(FancyBboxPatch((x, 2.5), w, 20, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=color, ec="#c9c8c3", lw=1))  # fmt: skip
        ax.text(x + w / 2, 18.5, title, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=INK)  # fmt: skip
        ax.text(x + w / 2, 10, body, ha="center", va="center", fontsize=7.5, color=INK2,
                linespacing=1.4)  # fmt: skip
    for (x0, *_), w0, (x1, *_) in zip(boxes, widths, boxes[1:], strict=False):
        ax.add_patch(FancyArrowPatch((x0 + w0 + 0.6, 13), (x1 - 0.6, 13), arrowstyle="-|>",
                                     mutation_scale=12, color=INK2, lw=1.4))  # fmt: skip
    save(fig, "pipeline")


def e2e_table() -> pd.DataFrame | None:
    """Track C: retrieved top-20 (pooled corpus) -> compressor -> main reader, with the paired
    bootstrap difference to the uncompressed top-20 and the LITM closed-book row for reference."""
    path = RESULTS_DIR / "e2e" / "summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df.n == 500]
    df = df.sort_values("date").drop_duplicates(["method", "params", "llm_model"], keep="last")
    df["label"] = [label(m, p) for m, p in zip(df.method, df.params, strict=True)]
    rows, lines = [], []
    # main reader first, then PISCO against Mistral reading the same top-20 as text
    blocks = [(None, [MAIN_LLM], BASE),
              ("Mistral-7B nf4: PISCO so với văn bản trên cùng reader",
               ["mistral-7b-nf4", "mistral-7b-nf4+pisco"], "Văn bản (20 đoạn)")]  # fmt: skip
    for title, llms, base in blocks:
        sub = df[df.llm_model.isin(llms)]
        if base not in set(sub.label):
            continue
        acc = {}
        for r in sub.itertuples():
            rows_ = [json.loads(x) for x in open(RESULTS_DIR / "e2e" / f"{r.run_id}_gold-1.jsonl")]
            acc[r.label] = pd.DataFrame(rows_).set_index("qid")["accuracy"]
        if title:
            lines += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for r in sub.sort_values("comp_tokens", ascending=False).itertuples():
            if r.label == base:
                diff = "--"
            else:
                m, lo, hi = paired_bootstrap(acc[base], acc[r.label])
                diff = f"{m:+.1f} [{lo:.1f}; {hi:.1f}]".replace("-", "$-$")
            rows.append({"label": r.label, "reader": r.llm_model, "recall": 100 * r.ctx_recall,
                         "acc": 100 * r.accuracy, "tokens": r.comp_tokens, "ratio": r.ratio,
                         "diff": diff})  # fmt: skip
            lines.append(f"{tex_name(r.label)} & {100 * r.ctx_recall:.1f} & "
                         f"{100 * r.accuracy:.1f} & {r.comp_tokens:.0f} & "
                         f"{r.ratio:.1f}$\\times$ & {diff} \\\\")  # fmt: skip
    write_tex("e2e.tex", lines)
    return pd.DataFrame(rows)


def fig_soft_loss() -> None:
    """Training loss of the encode-once soft compressor (results/soft/run1_train.jsonl)."""
    f = RESULTS_DIR / "soft" / "run1_train.jsonl"
    if not f.exists():
        return
    d = pd.read_json(f, lines=True)
    d = d[d.step > 1]  # step 1 is a single batch, the rest are 10-step means
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(d.step, d.loss, color=C[0], lw=1, alpha=0.35)
    ax.plot(d.step, d.loss.rolling(10, min_periods=1).mean(), color=C[0], lw=2)
    ax.axvline(1000, color=GRAY, ls=":", lw=1.2)
    ax.text(1015, ax.get_ylim()[1] * 0.95, "resume từ bước 1000", va="top", fontsize=8,
            color=INK2)  # fmt: skip
    ax.set_xlabel("Bước huấn luyện (8 mẫu/bước)")
    ax.set_ylabel("Loss (nats/token đáp án)")
    save(fig, "soft_loss")


def copy_tables_to_slides() -> None:
    """Slides are a separate Overleaf project, so they get their own copy of the tables."""
    import shutil

    dst = ROOT / "docs" / "slide" / "tables"
    dst.mkdir(parents=True, exist_ok=True)
    for f in TAB.glob("*.tex"):
        shutil.copy(f, dst / f.name)


def main() -> None:
    fig_pipeline()
    fig_soft_loss()
    t_rec = recall_table()
    print(t_rec.round(1).to_string())
    lat = latency_table()
    if lat is not None:
        print(lat.round(0).to_string())
    retrieval_table()
    if not (RESULTS_DIR / "compression" / "summary.csv").exists():
        return
    df = load("compression")
    t = main_table(df)
    t.round(2).to_csv(RESULTS_DIR / "compression" / "main_table.csv")
    write_latex(t)
    print(published_table(df).round(1).to_string())
    fig_position(df)
    fig_tradeoff(t)
    print(t.round(1).to_string())
    soft = soft_table()
    if soft is not None:
        print(soft.round(1).to_string())
    e2e = e2e_table()
    if e2e is not None:
        print(e2e.round(1).to_string())
    sig = significance(["Rerank top-5", "Rerank top-10", "LLMLingua-2 r=0.25",
                        "Rerank-10 → LLMLingua-2", "Provence τ=0.1", "OpenProvence τ=0.1",
                        "Truncate 730", "LongLLMLingua 730t", "BM25 top-5",
                        "Dense top-5"])  # fmt: skip
    if not sig.empty:
        sig.round(1).to_csv(RESULTS_DIR / "compression" / "significance.csv", index=False)
        print(sig.round(1).to_string())
    head = significance(["LongLLMLingua 730t"], base="Provence τ=0.1")
    if not head.empty:  # positive delta = LongLLMLingua better than Provence
        print("LongLLMLingua vs Provence τ=0.1\n", head.round(1).to_string())


if __name__ == "__main__":
    main()
    copy_tables_to_slides()
