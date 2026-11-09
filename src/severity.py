"""Large-loss severity modelling with extreme value theory.

Above a high threshold u, excesses Y = X - u are modelled by the Generalized
Pareto Distribution (Pickands-Balkema-de Haan theorem):
    S(y) = (1 + xi * y / beta) ** (-1 / xi),  y >= 0
"""
import numpy as np
from scipy.optimize import minimize


def mean_excess(losses, thresholds):
    """Empirical mean excess e(u) = E[X - u | X > u]; roughly linear in u for a GPD tail."""
    return np.array([(losses[losses > u] - u).mean() for u in thresholds])


def gpd_negloglik(params, y):
    xi, beta = params
    if beta <= 0:
        return np.inf
    z = 1 + xi * y / beta
    if np.any(z <= 0):
        return np.inf
    if abs(xi) < 1e-8:
        return len(y) * np.log(beta) + y.sum() / beta
    return len(y) * np.log(beta) + (1 + 1 / xi) * np.log(z).sum()


def fit_gpd(y):
    """Maximum likelihood GPD fit with standard errors from the observed information matrix."""
    start = np.array([0.1, y.mean()])
    res = minimize(gpd_negloglik, start, args=(y,), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20_000})
    xi, beta = res.x
    # Numerical Hessian of the negative log-likelihood
    h = np.array([1e-5, 1e-5 * beta])
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h[i], np.eye(2)[j] * h[j]
            H[i, j] = (gpd_negloglik(res.x + ei + ej, y) - gpd_negloglik(res.x + ei - ej, y)
                       - gpd_negloglik(res.x - ei + ej, y) + gpd_negloglik(res.x - ei - ej, y)) / (4 * h[i] * h[j])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return {"xi": xi, "beta": beta, "se_xi": se[0], "se_beta": se[1],
            "loglik": -res.fun, "n": len(y)}


def gpd_survival(y, xi, beta):
    return (1 + xi * np.asarray(y) / beta) ** (-1 / xi)


def gpd_quantile(p, xi, beta):
    return beta / xi * ((1 - np.asarray(p)) ** (-xi) - 1)


def hill(losses, k_values):
    """Hill estimator of the tail index xi using the k largest losses."""
    x = np.sort(losses)[::-1]
    return np.array([np.mean(np.log(x[:k])) - np.log(x[k]) for k in k_values])


def gpd_layer_expected(attach, limit, u, xi, beta):
    """E[min((X - attach)+, limit) | X > u] for a GPD excess over u, with attach >= u.

    Closed form of the integral of the survival function over the layer (xi != 1).
    """
    d = attach - u
    s = lambda y: (1 + xi * y / beta) ** (1 - 1 / xi)
    return beta / (1 - xi) * (s(d) - s(d + limit))
