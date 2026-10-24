"""Error analysis (T14): where does compression hurt or help, compared with the full context?

For each compressor and gold position, every question is paired with the full-context run:
  broken = full correct -> compressed wrong, split into "evidence lost" (answer no longer in the
           compressed context) and "evidence kept" (answer still there, the reader fails anyway);
  fixed  = full wrong -> compressed correct (less distraction).
Writes docs/report/tables/errors.tex and prints a few examples per category.

    uv run python scripts/error_analysis.py
"""

import textwrap

import pandas as pd
from make_figures import POS, load, per_question, tex_name, write_tex

from effrag.config import RESULTS_DIR

BASE = "Full (20 docs)"
SHOW = ["Provence τ=0.1", "LLMLingua-2 r=0.25", "Rerank top-5"]


def analyse() -> pd.DataFrame:
    df = load("compression")
    labels = [lab for lab in df.label.unique() if lab not in {BASE, "Closed-book", "Oracle"}]
    out = []
    for pos in POS:
        ref = per_question(BASE, pos)
        if ref is None:
            continue
        for lab in labels:
            q = per_question(lab, pos)
            if q is None:
                continue
            j = ref[["accuracy"]].join(q[["accuracy", "ctx_recall"]], rsuffix="_c", how="inner")
            broken = (j.accuracy == 1) & (j.accuracy_c == 0)
            out.append({
                "label": lab, "pos": pos, "n": len(j),
                "broken_lost": int((broken & (j.ctx_recall == 0)).sum()),
                "broken_kept": int((broken & (j.ctx_recall == 1)).sum()),
                "fixed": int(((j.accuracy == 0) & (j.accuracy_c == 1)).sum()),
                "wrong_kept": int(((j.accuracy_c == 0) & (j.ctx_recall == 1)).sum()),
            })  # fmt: skip
    return pd.DataFrame(out)


def write_table(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("label")[["n", "broken_lost", "broken_kept", "fixed", "wrong_kept"]].sum()
    for c in ("broken_lost", "broken_kept", "fixed", "wrong_kept"):
        g[c] = 100 * g[c] / g.n
    g["net"] = g.fixed - g.broken_lost - g.broken_kept
    g = g.sort_values("net", ascending=False)
    lines = [
        f"{tex_name(lab)} & {r.broken_lost:.1f} & {r.broken_kept:.1f} & {r.fixed:.1f} & "
        f"{r.net:+.1f} & {r.wrong_kept:.1f} \\\\"
        for lab, r in g.iterrows()
    ]
    write_tex("errors.tex", lines)
    return g


def examples(lab: str, pos: int = 9, k: int = 3) -> None:
    ref, q = per_question(BASE, pos), per_question(lab, pos)
    if ref is None or q is None:
        return
    j = ref[["accuracy", "prediction"]].join(q, rsuffix="_c", how="inner")
    cases = {
        "broken, evidence kept": j[(j.accuracy == 1) & (j.accuracy_c == 0) & (j.ctx_recall == 1)],
        "broken, evidence lost": j[(j.accuracy == 1) & (j.accuracy_c == 0) & (j.ctx_recall == 0)],
        "fixed": j[(j.accuracy == 0) & (j.accuracy_c == 1)],
    }
    print(f"\n===== {lab} @ gold {pos + 1}")
    for name, sub in cases.items():
        print(f"--- {name}: {len(sub)}")
        for qid, r in sub.head(k).iterrows():
            print(f"[{qid}] Q: {r.question} | gold: {r.answers}")
            print("   full:", textwrap.shorten(r.prediction, 160))
            print("   comp:", textwrap.shorten(r.prediction_c, 160))


def main() -> None:
    t = analyse()
    if t.empty:
        print("no LLM results yet")
        return
    t.to_csv(RESULTS_DIR / "compression" / "errors.csv", index=False)
    print(write_table(t).round(1).to_string())
    for lab in SHOW:
        examples(lab)


if __name__ == "__main__":
    main()
