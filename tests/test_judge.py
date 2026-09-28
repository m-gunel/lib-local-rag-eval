import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge_merge import weighted_kappa  # noqa: E402


def test_perfect_agreement_is_one():
    assert weighted_kappa([0, 1, 2, 3, 3], [0, 1, 2, 3, 3]) == pytest.approx(1.0)


def test_opposite_extremes_is_negative():
    assert weighted_kappa([0, 3, 0, 3], [3, 0, 3, 0]) < 0


def test_hand_computed_example():
    # a=[0,0,3,3], b=[0,1,3,3]: одно расхождение на 1 балл (вес 1/9).
    # po-взвешенное: do = (1/9)/4 = 1/36.
    # маргиналы: pa = [.5,0,0,.5], pb = [.25,.25,0,.5]
    # de = Σ w_ij pa_i pb_j = .5*(.25*0 + .25*1/9 + .5*1) + .5*(.25*1 + .25*4/9 + .5*0)
    de = 0.5 * (0.25 / 9 + 0.5) + 0.5 * (0.25 + 0.25 * 4 / 9)
    assert weighted_kappa([0, 0, 3, 3], [0, 1, 3, 3]) == pytest.approx(1 - (1 / 36) / de)
