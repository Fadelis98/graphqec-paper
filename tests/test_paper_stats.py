"""Table 1 statistics reproduce the published headline cell and the SI integer bounds."""

import numpy as np

from graphqec.benchmark import paper_stats as ps


def test_bb144_headline_cell():
    t = ps.table1_metrics(19_977_702_000, 110, 310, 12, 12)
    assert f"{t['block_round_event']:.2e}" == "4.59e-10"
    assert (f"{t['block_round_event_lo']:.2e}", f"{t['block_round_event_hi']:.2e}") == ("3.81e-10", "5.53e-10")
    assert f"{t['marginal_round_event']:.2e}" == "1.08e-10"
    assert t["marginal_interval_method"] == "none_per_shot_vectors_not_preserved"


def test_integer_deff_bounds_and_conservative_interval():
    lo, hi = ps.integer_deff_bounds(19_977_702_000, 110, 310, 12)
    assert (round(lo, 3), round(hi, 3)) == (2.871, 8.684)
    c_lo, c_hi, _, _ = ps.deff_wilson(19_977_702_000, 12, 310, hi * 310 / 19_977_702_000)
    assert (f"{float(ps.per_round_event(c_lo, 12)) * 1e10:.2f}", f"{float(ps.per_round_event(c_hi, 12)) * 1e10:.2f}") \
        == ("0.78", "1.49")


def test_k1_marginal_equals_block_and_var_x():
    t = ps.table1_metrics(10_000, 37, 37, 1, 5)
    assert t["marginal_round_event_lo"] == t["block_round_event_lo"]
    errors = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
    assert np.isclose(ps.var_x_from_errors(errors, 10), (4 + 1) / 10 - (3 / 10) ** 2)


def test_zero_failure_bound():
    t = ps.table1_metrics(201_200_000, 0, 0, 12, 12)
    assert t["block_zero_ub_round_event"] is not None and t["block_round_event"] == 0.0
