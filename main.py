"""
main.py
=======
Entry point — runs the full analysis pipeline and saves all figures + report.

Usage:
    python main.py
"""

import os
import sys
import time
import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pricing_engine import BlackScholes, MonteCarloPricer, OptionParams
from visualisation import (
    plot_gbm_paths,
    plot_terminal_distribution,
    plot_convergence,
    plot_price_surface,
    plot_summary_dashboard,
)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Define base contract
# ============================================================
PARAMS_CALL = OptionParams(S0=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
PARAMS_PUT  = OptionParams(S0=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")

N_MC = 500_000   # final pricing sims


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_pricing_report():
    section("BLACK-SCHOLES ANALYTICAL PRICES")
    bs_call = BlackScholes.price(PARAMS_CALL)
    bs_put  = BlackScholes.price(PARAMS_PUT)
    greeks  = BlackScholes.greeks(PARAMS_CALL)

    print(f"  Call price : {bs_call:.6f}")
    print(f"  Put  price : {bs_put:.6f}")
    print(f"\n  Call Greeks (S0=100, K=100, T=1y, r=5%, σ=20%):")
    for g, v in greeks.items():
        print(f"    {g:6s} = {v:+.6f}")

    # Parity check
    parity_lhs = bs_call - bs_put
    parity_rhs = PARAMS_CALL.S0 - PARAMS_CALL.K * np.exp(-PARAMS_CALL.r * PARAMS_CALL.T)
    print(f"\n  Put-Call Parity check:  C-P = {parity_lhs:.6f}  |  S-Ke^{{-rT}} = {parity_rhs:.6f}")
    print(f"  Error: {abs(parity_lhs - parity_rhs):.2e}  ✓" if abs(parity_lhs - parity_rhs) < 1e-8
          else "  WARNING: parity violated!")

    section("MONTE CARLO PRICING")
    pricer = MonteCarloPricer(seed=42)

    for params, label in [(PARAMS_CALL, "Call"), (PARAMS_PUT, "Put")]:
        t0  = time.perf_counter()
        res = pricer.price(params, n=N_MC, antithetic=True)
        elapsed = time.perf_counter() - t0
        bs  = BlackScholes.price(params)

        print(f"\n  European {label}  (N={res.n_simulations:,}, antithetic variates)")
        print(f"    MC price   : {res.price:.6f}")
        print(f"    BS price   : {bs:.6f}")
        print(f"    Error      : {abs(res.price - bs):.6f}  ({abs(res.price-bs)/bs*100:.3f}%)")
        print(f"    Std error  : {res.stderr:.6f}")
        print(f"    95% CI     : [{res.ci_lower:.6f}, {res.ci_upper:.6f}]")
        print(f"    Runtime    : {elapsed*1000:.1f} ms")

    section("SENSITIVITY TABLE  (varying σ)")
    print(f"\n  {'σ':>6}  {'BS Call':>9}  {'MC Call':>9}  {'Error':>9}  {'95% CI width':>14}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*14}")
    for sigma in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        p  = OptionParams(100, 100, 1.0, 0.05, sigma, "call")
        bs = BlackScholes.price(p)
        r  = pricer.price(p, n=200_000, antithetic=True)
        ci_w = r.ci_upper - r.ci_lower
        print(f"  {sigma:>6.0%}  {bs:>9.4f}  {r.price:>9.4f}  "
              f"{abs(r.price-bs):>9.5f}  {ci_w:>14.5f}")


def run_visualisations():
    section("GENERATING FIGURES")

    figs = [
        ("1_gbm_paths.png",             lambda: plot_gbm_paths(PARAMS_CALL, n_paths=40)),
        ("2_terminal_distribution.png", lambda: plot_terminal_distribution(PARAMS_CALL, n=300_000)),
        ("3_convergence.png",           lambda: plot_convergence(PARAMS_CALL)),
        ("4_price_surface.png",         lambda: plot_price_surface(PARAMS_CALL)),
        ("5_dashboard.png",             lambda: plot_summary_dashboard(PARAMS_CALL, PARAMS_PUT, n=200_000)),
    ]

    for fname, fn in figs:
        path = os.path.join(OUT, fname)
        print(f"  Generating {fname} ...", end="", flush=True)
        t0 = time.perf_counter()
        fig = fn()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  done  ({time.perf_counter()-t0:.1f}s)")

    print(f"\n  All figures saved to ./{OUT}/")


if __name__ == "__main__":
    run_pricing_report()
    run_visualisations()
    print("\n  Pipeline complete.\n")