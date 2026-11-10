"""Exposure rating of a property per-risk excess-of-loss layer with MBBEFD exposure curves.

Bernegger (1997) one-parameter family: c = 1.5, 2, 3, 4 correspond to the Swiss Re
Y1-Y4 curves and c = 5 approximates the Lloyd's industrial curve.
    b(c) = exp(3.1 - 0.15 (1 + c) c),  g(c) = exp((0.78 + 0.12 c) c)
    G(x) = ln[((g - 1) b + (1 - g b) b**x) / (1 - b)] / ln(g b),  0 <= x <= 1
G(x) is the share of expected loss below a deductible equal to a fraction x of the
sum insured; 1 / g is the probability of a total loss.
"""
import numpy as np


def mbbefd_params(c):
    b = np.exp(3.1 - 0.15 * (1 + c) * c)
    g = np.exp((0.78 + 0.12 * c) * c)
    return b, g


def exposure_curve(x, c):
    b, g = mbbefd_params(c)
    x = np.clip(np.asarray(x, dtype=float), 0, 1)
    return np.log(((g - 1) * b + (1 - g * b) * b ** x) / (1 - b)) / np.log(g * b)


def exposure_rate(profile, attach, limit, c, loss_ratio):
    """Expected layer loss for a risk profile (bands of sum insured and premium).

    For each band: premium x loss ratio x [G((attach + limit) / SI) - G(attach / SI)].
    """
    si = profile["sum_insured"].values
    share = exposure_curve((attach + limit) / si, c) - exposure_curve(attach / si, c)
    return float((profile["premium"].values * loss_ratio * share).sum())


def check_curve(c, h=1e-7):
    """Non-trivial consistency checks of the MBBEFD exposure curve (Bernegger, 1997).

    G'(0) = 1 / mu, with mu = ln(gb)(1 - b) / (ln(b)(1 - gb)) the mean loss degree, and
    G'(1) / G'(0) = p = 1 / g, the probability of a total loss.
    """
    b, g = mbbefd_params(c)
    d0 = (exposure_curve(h, c) - exposure_curve(0.0, c)) / h
    d1 = (exposure_curve(1.0, c) - exposure_curve(1.0 - h, c)) / h
    mu = np.log(g * b) * (1 - b) / (np.log(b) * (1 - g * b))
    return {"curve_c": c, "G(0)": float(exposure_curve(0.0, c)), "G(1)": float(exposure_curve(1.0, c)),
            "1/G'(0)": 1 / d0, "mean_loss_degree_formula": mu,
            "G'(1)/G'(0)": d1 / d0, "total_loss_prob_1_over_g": 1 / g}
