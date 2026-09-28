"""Метрики ранжирования по документам (методика, раздел «Метрики»).

Прогон — упорядоченный список документов из ответа индексатора (как есть,
без пересортировки по score). Разметка — {qid: {doc_id: оценка 0..3}}.

Правила:
- позиции после конца списка считаются нерелевантными;
- для Hit, MRR и Recall релевантен документ с оценкой ≥ 2;
- nDCG@10 — линейный gain rel/log2(i+1), IDCG по всем оценённым документам;
- в метрики ранжирования входят только запросы, у которых есть документ с
  оценкой ≥ 2 (запросы без ответа идут только в подбор порога).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence

REL_THRESHOLD = 2
K = 10

Run = Mapping[str, Sequence[str]]
Qrels = Mapping[str, Mapping[str, int]]


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], judged: Mapping[str, int], k: int = K) -> float:
    gains = [judged.get(d, 0) for d in ranked[:k]]
    ideal = sorted(judged.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def relevant(judged: Mapping[str, int]) -> set[str]:
    return {d for d, g in judged.items() if g >= REL_THRESHOLD}


def hit_at_k(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    rel = relevant(judged)
    return 1.0 if any(d in rel for d in ranked[:k]) else 0.0


def mrr_at_k(ranked: Sequence[str], judged: Mapping[str, int], k: int = K) -> float:
    rel = relevant(judged)
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: Sequence[str], judged: Mapping[str, int], k: int = K) -> float:
    rel = relevant(judged)
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


METRICS: dict[str, Callable[[Sequence[str], Mapping[str, int]], float]] = {
    "nDCG@10": lambda r, j: ndcg_at_k(r, j, 10),
    "Hit@1": lambda r, j: hit_at_k(r, j, 1),
    "Hit@10": lambda r, j: hit_at_k(r, j, 10),
    "MRR@10": lambda r, j: mrr_at_k(r, j, 10),
    "Recall@10": lambda r, j: recall_at_k(r, j, 10),
}


def dedupe(ranked: Sequence[str]) -> list[str]:
    """Документ считается один раз, на первой позиции (ответ индексатора уже
    сгруппирован по файлу, но на всякий случай)."""
    seen, out = set(), []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def answerable(qrels: Qrels) -> list[str]:
    return sorted(q for q, j in qrels.items() if relevant(j))


def per_query(run: Run, qrels: Qrels, qids: Sequence[str] | None = None) -> dict[str, dict[str, float]]:
    """Значения метрик по каждому запросу. Запрос без ответа индексатора
    (нет в прогоне) — пустой список, то есть нули."""
    qids = list(qids) if qids is not None else answerable(qrels)
    out = {}
    for q in qids:
        ranked = dedupe(run.get(q, []))
        out[q] = {name: fn(ranked, qrels[q]) for name, fn in METRICS.items()}
    return out


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(
    per_q: Mapping[str, Mapping[str, float]],
    metric: str,
    clusters: Mapping[str, str] | None = None,
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260925,
) -> tuple[float, float, float]:
    """Среднее и 95% доверительный интервал бутстрэпом.

    clusters: qid → id кластера (базовый вопрос). Варианты одного вопроса
    зависимы, поэтому ресэмплируются кластеры целиком.
    """
    qids = sorted(per_q)
    if not qids:
        return float("nan"), float("nan"), float("nan")
    groups: dict[str, list[str]] = {}
    for q in qids:
        groups.setdefault(clusters.get(q, q) if clusters else q, []).append(q)
    keys = sorted(groups)
    rnd = random.Random(seed)
    point = mean([per_q[q][metric] for q in qids])
    stats = []
    for _ in range(n):
        sample = [q for _ in keys for q in groups[keys[rnd.randrange(len(keys))]]]
        stats.append(mean([per_q[q][metric] for q in sample]))
    stats.sort()
    lo = stats[int(n * alpha / 2)]
    hi = stats[min(n - 1, int(n * (1 - alpha / 2)))]
    return point, lo, hi


def roc_auc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """Вероятность, что сигнал у запроса с ответом выше, чем у запроса без ответа."""
    if not pos_scores or not neg_scores:
        return float("nan")
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos_scores) * len(neg_scores))


def pick_threshold(
    pos_scores: Sequence[float], neg_scores: Sequence[float], max_loss: float = 0.05
) -> dict[str, float]:
    """Порог τ: теряем (скрываем выдачу) не больше max_loss запросов с ответом,
    а долю пустых выдач на запросах без ответа максимизируем.

    Выдача показывается, если сигнал ≥ τ.
    """
    candidates = sorted(set(pos_scores) | set(neg_scores))
    best = {"tau": float("-inf"), "loss": 0.0, "empty_on_noanswer": 0.0}
    for tau in candidates:
        loss = sum(s < tau for s in pos_scores) / len(pos_scores)
        if loss > max_loss:
            break
        empty = sum(s < tau for s in neg_scores) / len(neg_scores)
        if empty >= best["empty_on_noanswer"]:
            best = {"tau": tau, "loss": loss, "empty_on_noanswer": empty}
    return best
