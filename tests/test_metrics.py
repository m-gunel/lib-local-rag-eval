"""Метрики сверяются со значениями, посчитанными вручную."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import (  # noqa: E402
    answerable,
    bootstrap_ci,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    per_query,
    pick_threshold,
    recall_at_k,
    roc_auc,
)

J = {"a": 3, "b": 2, "c": 1, "z": 0}


def test_ndcg_perfect_order_is_one():
    assert ndcg_at_k(["a", "b", "c"], J) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # выдача: x(0), b(2), a(3)
    dcg = 0 / math.log2(2) + 2 / math.log2(3) + 3 / math.log2(4)
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(["x", "b", "a"], J) == pytest.approx(dcg / idcg)


def test_ndcg_short_list_counts_missing_as_irrelevant():
    # один документ в ответе вместо десяти — остальные позиции нули
    assert ndcg_at_k(["c"], J) == pytest.approx(1 / (3 + 2 / math.log2(3) + 1 / math.log2(4)))


def test_hit_and_mrr_use_threshold_two():
    # c имеет оценку 1 — не релевантен для Hit/MRR
    assert hit_at_k(["c", "x"], J, 1) == 0.0
    assert hit_at_k(["c", "b"], J, 10) == 1.0
    assert mrr_at_k(["c", "x", "b"], J) == pytest.approx(1 / 3)
    assert mrr_at_k(["c", "x"], J) == 0.0


def test_recall_counts_only_grade_two_plus():
    assert recall_at_k(["b", "c"], J) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], J) == pytest.approx(1.0)


def test_cutoff_at_ten():
    ranked = [f"x{i}" for i in range(10)] + ["a"]
    assert hit_at_k(ranked, J, 10) == 0.0
    assert mrr_at_k(ranked, J) == 0.0


def test_answerable_excludes_no_answer_queries():
    qrels = {"q1": J, "q2": {"c": 1}, "q3": {}}
    assert answerable(qrels) == ["q1"]


def test_per_query_missing_run_gives_zeros_and_dedupes():
    qrels = {"q1": J, "q2": {"b": 2}}
    pq = per_query({"q1": ["a", "a", "b"]}, qrels)
    assert pq["q1"]["Hit@1"] == 1.0
    assert pq["q1"]["Recall@10"] == 1.0
    assert pq["q2"] == {"nDCG@10": 0.0, "Hit@1": 0.0, "Hit@10": 0.0, "MRR@10": 0.0, "Recall@10": 0.0}


def test_bootstrap_ci_contains_mean_and_clusters_resample_whole_groups():
    pq = {f"q{i}": {"m": float(i % 2)} for i in range(40)}
    point, lo, hi = bootstrap_ci(pq, "m", n=500)
    assert point == pytest.approx(0.5)
    assert lo < point < hi
    clusters = {f"q{i}": f"c{i // 2}" for i in range(40)}  # пары 0/1 — среднее кластера всегда 0.5
    point, lo, hi = bootstrap_ci(pq, "m", clusters=clusters, n=500)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_roc_auc():
    assert roc_auc([3, 4], [1, 2]) == 1.0
    assert roc_auc([1, 2], [3, 4]) == 0.0
    assert roc_auc([1], [1]) == 0.5


def test_pick_threshold_respects_loss_budget():
    pos = [float(x) for x in range(20, 40)]  # 20 запросов с ответом
    neg = [float(x) for x in range(0, 25)]
    best = pick_threshold(pos, neg, max_loss=0.05)
    # терять можно не больше 1 из 20 → τ = 21
    assert best["tau"] == 21.0
    assert best["loss"] == pytest.approx(0.05)
    assert best["empty_on_noanswer"] == pytest.approx(21 / 25)
