"""Per-risk excess-of-loss reinsurance pricing on the Danish fire large-loss data (1980-1990)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, genpareto

from src.exposure_rating import check_curve, exposure_curve, exposure_rate
from src.pricing import (burning_cost, fit_frequency, reinstatement_pricing,
                         simulate_annual_layer_losses, technical_premium)
from src.severity import (fit_gpd, gpd_layer_expected, gpd_quantile, gpd_survival, hill, mean_excess)

ROOT = Path(__file__).parent
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)
NAVY, RED, GREY, GOLD = "#1F3864", "#C0392B", "#7F8C8D", "#D4A017"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

U = 10.0                                         # GPD threshold, million DKK
LAYERS = [(10, 10), (20, 30), (50, 50), (100, 150)]   # (attachment, limit), million DKK
N_YEARS, SEED = 1_000_000, 2026
COC, EXPENSES = 0.10, 0.10


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(ROOT / "data" / "danish_fire_losses.csv", parse_dates=["date"])
    x, years = df["loss_mdkk"].values, df["date"].dt.year.values
    n_years_data = len(np.unique(years))

    # 1. Tail diagnostics and GPD fit
    y = x[x > U] - U
    gpd = fit_gpd(y)
    c_sp, _, b_sp = genpareto.fit(y, floc=0)
    expo_ll = -len(y) * np.log(y.mean()) - len(y)
    lr_stat = 2 * (gpd["loglik"] - expo_ll)
    severity = pd.DataFrame([
        ["Exceedances over 10m", gpd["n"], np.nan, 109],
        ["xi (tail index)", gpd["xi"], c_sp, 0.497],
        ["beta (scale)", gpd["beta"], b_sp, 6.98],
        ["SE of xi", gpd["se_xi"], np.nan, np.nan],
        ["LR test xi = 0 (exponential): statistic", lr_stat, np.nan, np.nan],
        ["LR test p-value", chi2.sf(lr_stat, 1), np.nan, np.nan],
    ], columns=["quantity", "this_project", "scipy", "published_mcneil_1997"])
    severity.to_csv(TABLES / "severity_fit.csv", index=False)

    # 2. Frequency of losses above the threshold
    counts = pd.Series(years[x > U]).value_counts().reindex(np.unique(years), fill_value=0).values
    freq = fit_frequency(counts)
    pd.DataFrame([freq]).to_csv(TABLES / "frequency_fit.csv", index=False)

    def sampler(n, rng_):
        return U + genpareto.ppf(rng_.random(n), gpd["xi"], scale=gpd["beta"])

    # 3. Layer pricing: burning cost vs model (analytic and Monte Carlo)
    rows, sims = [], {}
    for a, l in LAYERS:
        bc, bc_se, _ = burning_cost(x, years, a, l)
        analytic = freq["mean"] * gpd_layer_expected(a, l, U, gpd["xi"], gpd["beta"])
        ann = simulate_annual_layer_losses(freq, sampler, a, l, N_YEARS, rng)
        sims[(a, l)] = ann
        tp = technical_premium(ann, l, COC, EXPENSES)
        lo = freq["mean"] * gpd_layer_expected(a, l, U, gpd["xi"] - gpd["se_xi"], gpd["beta"])
        hi = freq["mean"] * gpd_layer_expected(a, l, U, gpd["xi"] + gpd["se_xi"], gpd["beta"])
        rows.append({
            "layer": f"{l:g} xs {a:g}", "burning_cost": bc, "burning_cost_se": bc_se,
            "model_expected_loss_analytic": analytic, "model_expected_loss_mc": ann.mean(),
            "mc_se": ann.std(ddof=1) / np.sqrt(N_YEARS),
            "expected_loss_xi_minus_1se": lo, "expected_loss_xi_plus_1se": hi,
            "losses_in_history": int((x > a).sum()), **tp,
        })
    layers = pd.DataFrame(rows)
    layers.to_csv(TABLES / "layer_pricing.csv", index=False)

    # 4. Reinstatements on the 30 xs 20 layer
    ann = sims[(20, 30)]
    reinst = pd.DataFrame([{"reinstatements": k, "rate": r, **reinstatement_pricing(ann, 30, k, r)}
                           for k, r in [(0, 1.0), (1, 1.0), (2, 1.0), (2, 0.5)]])
    unlimited = ann.mean()
    reinst.insert(0, "unlimited_free_cover_expected_loss", unlimited)
    reinst.to_csv(TABLES / "reinstatements_30xs20.csv", index=False)

    # 5. Exposure rating (illustrative risk profile, million DKK)
    profile = pd.DataFrame({"sum_insured": [5, 10, 20, 50, 100, 200],
                            "premium": [60.0, 45.0, 30.0, 20.0, 10.0, 5.0]})
    exposure = pd.DataFrame([{"curve_c": c, "label": lab,
                              "expected_loss_30xs20": exposure_rate(profile, 20, 30, c, 0.60),
                              "total_loss_probability": 1 / np.exp((0.78 + 0.12 * c) * c)}
                             for c, lab in [(2, "Swiss Re Y2"), (3, "Swiss Re Y3"), (4, "Swiss Re Y4"),
                                            (5, "Lloyd's industrial (approx.)")]])
    exposure.to_csv(TABLES / "exposure_rating_illustrative.csv", index=False)
    profile.to_csv(TABLES / "illustrative_risk_profile.csv", index=False)
    curve_checks = pd.DataFrame([check_curve(c) for c in (1.5, 2, 3, 4, 5)])
    curve_checks.to_csv(TABLES / "exposure_curve_checks.csv", index=False)

    # 6. Charts
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    us = np.linspace(1, 60, 120)
    axes[0].plot(us, mean_excess(x, us), color=NAVY)
    axes[0].axvline(U, color=RED, ls="--", lw=1, label="Chosen threshold: 10m")
    axes[0].set_xlabel("Threshold u (million DKK)"); axes[0].set_ylabel("Mean excess e(u)")
    axes[0].set_title("Mean excess plot: roughly linear above ~10m (GPD tail)")
    axes[0].legend(frameon=False, fontsize=8)
    ks = np.arange(20, 501, 5)
    axes[1].plot(ks, hill(x, ks), color=NAVY, label="Hill estimator")
    axes[1].axhline(gpd["xi"], color=RED, ls="--", lw=1, label=f"GPD xi above 10m: {gpd['xi']:.2f}")
    axes[1].set_xlabel("Number of largest losses k"); axes[1].set_ylabel("Tail index xi")
    axes[1].set_title("Hill plot")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "tail_diagnostics.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ys = np.sort(y)
    p = (np.arange(1, len(ys) + 1) - 0.5) / len(ys)
    axes[0].scatter(gpd_quantile(p, gpd["xi"], gpd["beta"]), ys, s=12, color=NAVY)
    lim = max(ys.max(), gpd_quantile(p[-1], gpd["xi"], gpd["beta"]))
    axes[0].plot([0, lim], [0, lim], color=RED, lw=1)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("GPD quantile"); axes[0].set_ylabel("Observed excess over 10m")
    axes[0].set_title("QQ plot of excesses (log scale)")
    emp_s = 1 - (np.arange(1, len(ys) + 1) - 0.5) / len(ys)
    axes[1].scatter(ys + U, emp_s, s=12, color=NAVY, label="Empirical")
    grid = np.linspace(0, ys.max() * 1.5, 400)
    axes[1].plot(grid + U, gpd_survival(grid, gpd["xi"], gpd["beta"]), color=RED, label="GPD fit")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Loss (million DKK)"); axes[1].set_ylabel("P(X > x | X > 10m)")
    axes[1].set_title("Tail survival function")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "gpd_fit.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(len(layers))
    ax.bar(idx - 0.2, layers["burning_cost"], width=0.4, color=GREY,
           yerr=[np.minimum(1.96 * layers["burning_cost_se"], layers["burning_cost"]), 1.96 * layers["burning_cost_se"]],
           capsize=3, label="Burning cost (history), 95% CI floored at 0")
    ax.bar(idx + 0.2, layers["model_expected_loss_analytic"], width=0.4, color=NAVY,
           yerr=[layers["model_expected_loss_analytic"] - layers["expected_loss_xi_minus_1se"],
                 layers["expected_loss_xi_plus_1se"] - layers["model_expected_loss_analytic"]],
           capsize=3, label="GPD model, range for xi +/- 1 SE")
    ax.set_xticks(idx, layers["layer"])
    ax.set_ylabel("Expected annual layer loss (million DKK)")
    ax.set_title("Experience (burning cost) vs model pricing by layer")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "layer_pricing.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    grid = np.linspace(0, 1, 200)
    for (c, lab), col in zip([(2, "Y2 (c=2)"), (3, "Y3 (c=3)"), (4, "Y4 (c=4)"), (5, "Lloyd's (c=5)")],
                             [NAVY, RED, GOLD, GREY]):
        ax.plot(grid, exposure_curve(grid, c), color=col, label=lab)
    ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls=":")
    ax.set_xlabel("Deductible as a share of sum insured"); ax.set_ylabel("Share of expected loss retained")
    ax.set_title("MBBEFD exposure curves")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "exposure_curves.png", dpi=150); plt.close(fig)

    # 7. Console summary
    pd.set_option("display.width", 220)
    print(f"Losses: {len(x)} over {n_years_data} years; exceedances over {U:g}m: {len(y)}")
    print("\nSeverity\n" + severity.round(4).to_string(index=False))
    print("\nFrequency\n" + pd.DataFrame([freq]).round(3).to_string(index=False))
    cols = ["layer", "burning_cost", "model_expected_loss_analytic", "model_expected_loss_mc", "mc_se",
            "expected_loss_xi_minus_1se", "expected_loss_xi_plus_1se", "losses_in_history",
            "prob_any_loss", "var_99_5", "technical_premium", "rate_on_line", "payback_years"]
    print("\nLayers\n" + layers[cols].round(3).to_string(index=False))
    print("\nReinstatements 30 xs 20\n" + reinst.round(4).to_string(index=False))
    print("\nExposure rating (illustrative)\n" + exposure.round(3).to_string(index=False))
    print("\nExposure curve checks\n" + curve_checks.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
