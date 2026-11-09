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
