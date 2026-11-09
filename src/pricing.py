"""Excess-of-loss pricing: frequency, burning cost, collective risk model, reinstatements, loadings."""
import numpy as np


def fit_frequency(counts):
    """Poisson and negative binomial (method of moments) fits to annual claim counts."""
    mean, var = counts.mean(), counts.var(ddof=1)
    out = {"mean": mean, "variance": var, "dispersion": var / mean}
    if var > mean:
        p = mean / var
        out.update({"nb_r": mean * p / (1 - p), "nb_p": p})
    return out


def layer_loss(x, attach, limit):
    return np.clip(x - attach, 0, limit)


def burning_cost(losses, years, attach, limit):
    """Average annual layer loss from the historical (as-if) losses."""
    per_year = {y: layer_loss(losses[years == y], attach, limit).sum() for y in np.unique(years)}
    vals = np.array(list(per_year.values()))
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)), per_year


def simulate_annual_layer_losses(freq, sev_sampler, attach, limit, n_years, rng, use_nb=True):
    """Collective risk model: annual aggregate loss to a per-risk layer, by Monte Carlo."""
    if use_nb and "nb_r" in freq:
        counts = rng.negative_binomial(freq["nb_r"], freq["nb_p"], n_years)
    else:
        counts = rng.poisson(freq["mean"], n_years)
    sev = sev_sampler(counts.sum(), rng)
    per_claim = layer_loss(sev, attach, limit)
    year_idx = np.repeat(np.arange(n_years), counts)
    return np.bincount(year_idx, weights=per_claim, minlength=n_years)


def reinstatement_pricing(annual_loss, limit, n_reinstatements, rate=1.0):
    """Pure premium of a layer with paid reinstatements, pro rata to amount.

    Recoveries are capped at (k + 1) x limit; each reinstatement costs rate x P x (amount / limit).
    The pure premium P solves  P + rate * P * E[min(L, k*l)] / l = E[min(L, (k+1) l)].
    """
    covered = np.minimum(annual_loss, (n_reinstatements + 1) * limit)
    reinstated = np.minimum(annual_loss, n_reinstatements * limit)
    p = covered.mean() / (1 + rate * reinstated.mean() / limit)
    rip = rate * p * reinstated / limit
    return {"pure_premium": p, "expected_recoveries": covered.mean(),
            "expected_reinstatement_premium": rip.mean(),
            "check_net_zero": (p + rip - covered).mean()}


def technical_premium(annual_loss, limit, coc=0.10, expense_ratio=0.10):
    """Expected loss plus a cost-of-capital loading on the 99.5% VaR in excess of the mean,
    grossed up for brokerage and expenses."""
    el = annual_loss.mean()
    capital = np.quantile(annual_loss, 0.995) - el
    premium = (el + coc * capital) / (1 - expense_ratio)
    return {"expected_loss": el, "sd": annual_loss.std(ddof=1), "var_99_5": np.quantile(annual_loss, 0.995),
            "capital": capital, "technical_premium": premium, "rate_on_line": premium / limit,
            "payback_years": limit / premium, "prob_any_loss": (annual_loss > 0).mean(),
            "loss_on_line": el / limit}
