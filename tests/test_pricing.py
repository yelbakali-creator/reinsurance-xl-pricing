"""Validation tests: GPD fit against McNeil (1997), closed-form layer loss, reinstatements, MBBEFD curves."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad

from src.exposure_rating import check_curve, exposure_curve
from src.pricing import reinstatement_pricing
from src.severity import fit_gpd, gpd_layer_expected, gpd_survival

X = pd.read_csv(Path(__file__).parents[1] / "data" / "danish_fire_losses.csv")["loss_mdkk"].values


def test_gpd_fit_matches_mcneil_1997():
    y = X[X > 10] - 10
    g = fit_gpd(y)
    assert len(y) == 109
    assert abs(g["xi"] - 0.497) < 0.002
    assert abs(g["beta"] - 6.98) < 0.01


def test_closed_form_layer_matches_numerical_integration():
    xi, beta = 0.497, 6.975
    closed = gpd_layer_expected(20, 30, 10, xi, beta)
    numeric = quad(lambda t: gpd_survival(t, xi, beta), 10, 40)[0]
    assert abs(closed - numeric) < 1e-10


def test_reinstatements_balance():
    rng = np.random.default_rng(0)
    losses = rng.exponential(20, 50_000)
    for k, rate in [(0, 1.0), (1, 1.0), (2, 0.5)]:
        res = reinstatement_pricing(losses, 30, k, rate)
        assert abs(res["check_net_zero"]) < 1e-8


def test_mbbefd_curves():
    for c in (1.5, 2, 3, 4, 5):
        chk = check_curve(c)
        assert abs(chk["G(0)"]) < 1e-12 and abs(chk["G(1)"] - 1) < 1e-12
        assert abs(chk["1/G'(0)"] / chk["mean_loss_degree_formula"] - 1) < 1e-12
        assert abs(chk["G'(1)/G'(0)"] / chk["total_loss_prob_1_over_g"] - 1) < 1e-12
        assert abs(chk["derivative_vs_finite_difference_at_0.5"]) < 1e-8
        g = exposure_curve(np.linspace(0, 1, 101), c)
        assert np.all(np.diff(g) >= -1e-12)                     # increasing
