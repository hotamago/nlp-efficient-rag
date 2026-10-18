"""Download the NQ-Open multi-document QA data from "Lost in the Middle" (Liu et al., TACL 2024).

Each example: a question, gold answers, and 20 Wikipedia passages retrieved by Contriever
(exactly one contains the answer). LongLLMLingua (Jiang et al., ACL 2024) reports on this setup.
"""

import argparse
import urllib.request
from pathlib import Path

BASE = "https://github.com/nelson-liu/lost-in-the-middle/raw/main/qa_data/20_total_documents"
GOLD_POSITIONS = [0, 4, 9, 14, 19]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/litm_nq20")
    parser.add_argument("--positions", type=int, nargs="+", default=GOLD_POSITIONS)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for pos in args.positions:
        name = f"nq-open-20_total_documents_gold_at_{pos}.jsonl.gz"
        dest = out / name
        if dest.exists():
            print(f"skip {dest}")
            continue
        print(f"download {name}")
        urllib.request.urlretrieve(f"{BASE}/{name}", dest)


if __name__ == "__main__":
    main()
