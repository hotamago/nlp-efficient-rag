"""Build report tables (LaTeX) and figures (PDF/PNG) from results/.

uv run python scripts/make_figures.py
"""

import json

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
    return {"full": "Full (20 docs)", "closedbook": "Closed-book", "oracle": "Oracle"}[method]


FAMILY = {
    "full": "Tham chiếu", "closedbook": "Tham chiếu", "oracle": "Tham chiếu",
    "truncate": "Prompt", "bm25_topk": "Tài liệu", "dense_topk": "Tài liệu",
    "rerank_topk": "Tài liệu", "llmlingua2": "Token", "longllmlingua": "Token",
    "chain": "Token", "provence": "Câu",
}  # fmt: skip


def load(track: str, n: int | None = 500) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / track / "summary.csv")
    if n is not None:
        df = df[df.n == n]
    # keep the latest run per (method, params, gold_pos)
    df = df.sort_values("date").drop_duplicates(["method", "params", "gold_pos"], keep="last")
    df["label"] = [label(m, p) for m, p in zip(df.method, df.params, strict=True)]
    df["family"] = df.method.map(FAMILY)
    # position-invariant references are run once and shown at every position
    extra = [
        df[(df.method == m)].assign(gold_pos=pos)
        for m in POS_INVARIANT
        for pos in POS
        if not ((df.method == m) & (df.gold_pos == pos)).any() and (df.method == m).any()
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


def tex_name(name: str) -> str:
    return name.replace("τ", "$\\tau$").replace("→", "$\\rightarrow$")


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
            tb = fmt(r.TB)
            if fam != "Tham chiếu" and abs(r.TB - best_tb) < 1e-9:
                tb = f"\\best{{{tb}}}"
            lines.append(
                f"\\quad {tex_name(name)} & {accs} & {tb} & {r.Token:.0f} & "
                f"{r['Nén']:.1f}$\\times$ & {r.Recall:.1f} & {fmt(r.ms, 0)} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    (TAB / "main_results.tex").write_text("\n".join(lines) + "\n")


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
    (TAB / "recall.tex").write_text("\n".join(lines) + "\n")
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


def published_table(df: pd.DataFrame, df200: pd.DataFrame) -> None:
    ours = pivot(df, "accuracy") * 100
    ours200 = pivot(df200, "accuracy") * 100 if not df200.empty else pd.DataFrame()
    lines = []
    for key, pub_name, pub in PUBLISHED:
        row = (
            ours.loc[key]
            if key in ours.index
            else ours200.loc[key]
            if key in ours200.index
            else None
        )
        vals = [fmt(row[p]) if row is not None else "--" for p in POS]
        lines.append(
            f"{pub_name} & "
            + " & ".join(f"{v:.1f}" for v in pub)
            + f" & {tex_name(key)} & "
            + " & ".join(vals)
            + " \\\\"
        )
    (TAB / "published.tex").write_text("\n".join(lines) + "\n")


def per_question(method_label: str, pos: int) -> pd.DataFrame | None:
    """Per-question rows of the latest LLM run for a config label at a gold position."""
    for path in sorted((RESULTS_DIR / "compression").glob(f"*_gold{pos}.jsonl"), reverse=True):
        rows = [json.loads(x) for x in path.open()]
        r0 = rows[0]
        run = r0["run_id"]
        summ = pd.read_csv(RESULTS_DIR / "compression" / "summary.csv")
        s = summ[(summ.run_id == run) & (summ.gold_pos == pos)]
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


def significance(labels: list[str], base: str = "Full (20 docs)") -> pd.DataFrame:
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


def fig_position(df: pd.DataFrame) -> None:
    acc = pivot(df, "accuracy") * 100
    series = ["Full (20 docs)", "Truncate 730", "Rerank top-5", "LLMLingua-2 r=0.25",
              "Rerank-10 → LLMLingua-2", "Provence τ=0.1"]  # fmt: skip
    series = [s for s in series if s in acc.index]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for ref, ls in (("Oracle", ":"), ("Closed-book", "--")):
        if ref in acc.index:
            y = acc.loc[ref].mean()
            ax.axhline(y, color=GRAY, ls=ls, lw=1.2)
            ax.text(2.06, y, f"{ref} {y:.1f}", va="center", fontsize=8, color=INK2)
    for i, s in enumerate(series):
        ax.plot(range(len(POS)), acc.loc[s], color=C[i], marker="o", ms=4, label=s)
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
    for name, r in t[t.family != "Tham chiếu"].iterrows():
        ax.annotate(name, (r.Token, r.TB), fontsize=6.5, color=INK2,
                    xytext=(4, 2), textcoords="offset points")  # fmt: skip
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
    (TAB / "latency.tex").write_text("\n".join(lines) + "\n")
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
            f"{s} & {100 * r['answer_hit@1']:.1f} & {100 * r['answer_hit@5']:.1f} & "
            f"{100 * r['answer_hit@20']:.1f} & {100 * r['gold_mrr@10']:.1f} & "
            f"{r.index_mb:.1f} & {r.query_ms:.2f} & {r.query_ms_single:.1f} \\\\"
        )
    (TAB / "retrieval.tex").write_text("\n".join(rows) + "\n")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh(df.index[::-1], 100 * df["answer_hit@5"][::-1], color=C[0], height=0.6)
    for y, v in enumerate(100 * df["answer_hit@5"][::-1]):
        ax.text(v + 0.5, y, f"{v:.1f}", va="center", fontsize=7, color=INK2)
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
        (85, "LLM", "Qwen2.5-7B\n(vLLM)\n→ câu trả lời", "#d7f1e6"),
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


def main() -> None:
    fig_pipeline()
    t_rec = recall_table()
    print(t_rec.round(1).to_string())
    lat = latency_table()
    if lat is not None:
        print(lat.round(0).to_string())
    retrieval_table()
    if not (RESULTS_DIR / "compression" / "summary.csv").exists():
        return
    df = load("compression")
    df200 = load("compression", n=200)
    t = main_table(df)
    t.round(2).to_csv(RESULTS_DIR / "compression" / "main_table.csv")
    write_latex(t)
    published_table(df, df200)
    fig_position(df)
    fig_tradeoff(t)
    print(t.round(1).to_string())
    sig = significance(["Rerank top-5", "Rerank top-10", "LLMLingua-2 r=0.25",
                        "Rerank-10 → LLMLingua-2", "Provence τ=0.1", "OpenProvence τ=0.1",
                        "Truncate 730"])  # fmt: skip
    if not sig.empty:
        sig.round(1).to_csv(RESULTS_DIR / "compression" / "significance.csv", index=False)
        print(sig.round(1).to_string())


if __name__ == "__main__":
    main()
