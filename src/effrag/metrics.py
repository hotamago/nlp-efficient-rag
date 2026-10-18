"""QA metrics. `best_subspan_em` is the accuracy used by Lost in the Middle and LongLLMLingua."""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def best_subspan_em(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(a) in pred for a in answers))


def exact_match(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(a) == pred for a in answers))


def f1(prediction: str, answers: list[str]) -> float:
    def _f1(p: str, g: str) -> float:
        pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
        common = sum((Counter(pt) & Counter(gt)).values())
        if common == 0:
            return 0.0
        prec, rec = common / len(pt), common / len(gt)
        return 2 * prec * rec / (prec + rec)

    return max(_f1(prediction, a) for a in answers)
