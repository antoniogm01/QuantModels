"""
visualisation.py
================
Publication-quality figures for the Monte Carlo option pricing project.
All plots use a consistent dark-quant aesthetic.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.lines import Line2D
from scipy.stats import norm

from pricing_engine import BlackScholes, MonteCarloPricer, OptionParams

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GRID     = "#21262d"
TEXT     = "#e6edf3"
ACCENT1  = "#58a6ff"   # blue  — MC estimate / call
ACCENT2  = "#f78166"   # coral — put / error
ACCENT3  = "#3fb950"   # green — BS analytical
ACCENT4  = "#d2a8ff"   # lavender — CI band
MUTED    = "#8b949e"


def _apply_base_style(fig, axes_flat):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_flat:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.6, linestyle="--", alpha=0.7)


# ---------------------------------------------------------------------------
# 1. GBM paths
# ---------------------------------------------------------------------------

def plot_gbm_paths(params: OptionParams, n_paths: int = 30, n_steps: int = 252,
                   save_path: str | None = None):
    pricer = MonteCarloPricer(seed=7)
    paths  = pricer.simulate_full_paths(params, n_paths=n_paths, n_steps=n_steps)
    t      = np.linspace(0, params.T, n_steps + 1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _apply_base_style(fig, [ax])

    # faint background paths
    for i in range(n_paths):
        color = ACCENT1 if paths[i, -1] > params.K else ACCENT2
        ax.plot(t, paths[i], color=color, alpha=0.35, linewidth=0.9)

    # mean path
    ax.plot(t, paths.mean(axis=0), color=ACCENT3, linewidth=2.2,
            label="Ensemble mean", zorder=5)

    # strike & spot markers
    ax.axhline(params.K, color=ACCENT4, linestyle=":", linewidth=1.4,
               label=f"Strike  K = {params.K:.0f}", zorder=4)
    ax.axhline(params.S0, color=MUTED,  linestyle="--", linewidth=1.0,
               label=f"Spot  S₀ = {params.S0:.0f}", zorder=4)

    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("Stock price  S(t)", fontsize=11)
    ax.set_title("Geometric Brownian Motion — Risk-Neutral Simulated Paths", fontsize=13)

    legend = ax.legend(fontsize=9, framealpha=0, labelcolor=TEXT)
    _color_legend(legend)

    # annotation box
    info = (
        f"S₀={params.S0}  K={params.K}  T={params.T}y\n"
        f"r={params.r:.1%}  σ={params.sigma:.1%}  N={n_paths} paths"
    )
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", color=MUTED,
            bbox=dict(facecolor=DARK_BG, edgecolor=GRID, boxstyle="round,pad=0.4"))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 2. Terminal price distribution
# ---------------------------------------------------------------------------

def plot_terminal_distribution(params: OptionParams, n: int = 200_000,
                                save_path: str | None = None):
    pricer = MonteCarloPricer(seed=99)
    result = pricer.price(params, n=n, store_paths=True)
    ST     = result.paths

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_base_style(fig, [ax])

    counts, bins, patches = ax.hist(
        ST, bins=120, density=True, color=ACCENT1, alpha=0.55, edgecolor="none"
    )
    # colour in-the-money region
    for patch, left in zip(patches, bins[:-1]):
        if params.option_type == "call" and left >= params.K:
            patch.set_facecolor(ACCENT3)
            patch.set_alpha(0.80)
        elif params.option_type == "put" and left <= params.K:
            patch.set_facecolor(ACCENT2)
            patch.set_alpha(0.80)

    # Overlay analytical log-normal PDF
    x = np.linspace(ST.min(), ST.max(), 600)
    mu_ln  = np.log(params.S0) + (params.r - 0.5 * params.sigma**2) * params.T
    sig_ln = params.sigma * np.sqrt(params.T)
    pdf    = norm.pdf(np.log(x), mu_ln, sig_ln) / x
    ax.plot(x, pdf, color=ACCENT4, linewidth=2.0, label="Log-normal PDF (analytical)")

    ax.axvline(params.K, color=ACCENT2, linestyle="--", linewidth=1.6,
               label=f"Strike K = {params.K:.0f}")
    ax.axvline(ST.mean(), color=ACCENT3, linestyle=":", linewidth=1.4,
               label=f"E[S_T] = {ST.mean():.2f}")

    ax.set_xlabel("Terminal stock price  S_T", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title("Risk-Neutral Terminal Price Distribution", fontsize=13)

    legend = ax.legend(fontsize=9, framealpha=0, labelcolor=TEXT)
    _color_legend(legend)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 3. Convergence analysis
# ---------------------------------------------------------------------------

def plot_convergence(params: OptionParams, save_path: str | None = None):
    pricer = MonteCarloPricer(seed=0)
    study  = pricer.convergence_study(params)

    n      = study["n_grid"]
    prices = study["prices"]
    ci_lo  = study["ci_lower"]
    ci_hi  = study["ci_upper"]
    bs     = study["bs_price"]
    error  = np.abs(prices - bs)

    fig = plt.figure(figsize=(13, 5.5))
    gs  = gridspec.GridSpec(1, 2, wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    _apply_base_style(fig, [ax1, ax2])

    # --- Left: price convergence ---
    ax1.fill_between(n, ci_lo, ci_hi, color=ACCENT4, alpha=0.25, label="95% CI")
    ax1.plot(n, prices, color=ACCENT1, linewidth=1.8, label="MC estimate")
    ax1.axhline(bs, color=ACCENT3, linewidth=1.6, linestyle="--",
                label=f"BS analytical = {bs:.4f}")

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of simulations  N", fontsize=11)
    ax1.set_ylabel("Option price", fontsize=11)
    ax1.set_title("Price Convergence vs N", fontsize=12)
    legend1 = ax1.legend(fontsize=9, framealpha=0, labelcolor=TEXT)
    _color_legend(legend1)

    # --- Right: absolute error on log-log ---
    ax2.scatter(n, error, color=ACCENT2, s=18, zorder=4, label="|MC − BS|")
    # 1/√N reference line
    ref_scale = error[5] * np.sqrt(n[5])
    ax2.plot(n, ref_scale / np.sqrt(n), color=MUTED, linewidth=1.2,
             linestyle=":", label=r"$O(N^{-1/2})$ reference")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of simulations  N", fontsize=11)
    ax2.set_ylabel("|MC price − BS price|", fontsize=11)
    ax2.set_title(r"Absolute Error  ($\log$–$\log$)", fontsize=12)
    legend2 = ax2.legend(fontsize=9, framealpha=0, labelcolor=TEXT)
    _color_legend(legend2)

    fig.suptitle(
        f"Monte Carlo Convergence Analysis  |  "
        f"S₀={params.S0}  K={params.K}  T={params.T}y  σ={params.sigma:.1%}",
        color=TEXT, fontsize=13, y=1.02
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 4. Volatility surface (price vs S0 and sigma)
# ---------------------------------------------------------------------------

def plot_price_surface(params: OptionParams, save_path: str | None = None):
    """Heat-map of analytical BS call price over (spot, vol) grid."""
    spots  = np.linspace(params.S0 * 0.6, params.S0 * 1.4, 60)
    sigmas = np.linspace(0.05, 0.60, 55)
    SS, SG = np.meshgrid(spots, sigmas)

    prices = np.zeros_like(SS)
    for i in range(SG.shape[0]):
        for j in range(SS.shape[1]):
            p2 = OptionParams(SS[i,j], params.K, params.T, params.r,
                              SG[i,j], params.option_type)
            prices[i, j] = BlackScholes.price(p2)

    fig, ax = plt.subplots(figsize=(9, 6))
    _apply_base_style(fig, [ax])

    im = ax.contourf(SS, SG, prices, levels=30, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Option price", color=TEXT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)

    # mark current params
    ax.scatter([params.S0], [params.sigma], color=ACCENT2, s=80,
               zorder=5, label=f"Current params (σ={params.sigma:.0%})")
    ax.axvline(params.K, color=ACCENT4, linestyle="--", linewidth=1.1,
               label=f"ATM (K={params.K})")

    ax.set_xlabel("Spot price  S₀", fontsize=11)
    ax.set_ylabel("Volatility  σ", fontsize=11)
    ax.set_title(f"Black-Scholes {params.option_type.capitalize()} Price Surface", fontsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    legend = ax.legend(fontsize=9, framealpha=0, labelcolor=TEXT)
    _color_legend(legend)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 5. Comprehensive summary dashboard
# ---------------------------------------------------------------------------

def plot_summary_dashboard(params_call: OptionParams, params_put: OptionParams,
                            n: int = 500_000, save_path: str | None = None):
    """4-panel dashboard: paths, distributions (call & put), convergence."""
    pricer = MonteCarloPricer(seed=42)

    paths  = pricer.simulate_full_paths(params_call, n_paths=40, n_steps=252)
    res_c  = pricer.price(params_call, n=n, store_paths=True)
    res_p  = pricer.price(params_put,  n=n, store_paths=True)
    study  = pricer.convergence_study(params_call)

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.32,
                            left=0.07, right=0.97, top=0.91, bottom=0.09)
    ax_paths = fig.add_subplot(gs[:, 0])   # left column — full height
    ax_hc    = fig.add_subplot(gs[0, 1])
    ax_hp    = fig.add_subplot(gs[0, 2])
    ax_conv  = fig.add_subplot(gs[1, 1:])
    _apply_base_style(fig, [ax_paths, ax_hc, ax_hp, ax_conv])

    t = np.linspace(0, params_call.T, paths.shape[1])

    # ---- GBM paths ----
    for i in range(paths.shape[0]):
        c = ACCENT1 if paths[i,-1] > params_call.K else ACCENT2
        ax_paths.plot(t, paths[i], color=c, alpha=0.3, linewidth=0.8)
    ax_paths.plot(t, paths.mean(axis=0), color=ACCENT3, linewidth=2.2)
    ax_paths.axhline(params_call.K, color=ACCENT4, linestyle=":", linewidth=1.4)
    ax_paths.set_xlabel("Time (years)", fontsize=10)
    ax_paths.set_ylabel("S(t)", fontsize=10)
    ax_paths.set_title("GBM Paths", fontsize=11)

    # ---- Call payoff histogram ----
    _payoff_hist(ax_hc, res_c.paths, params_call, ACCENT1, ACCENT3, "Call")

    # ---- Put payoff histogram ----
    _payoff_hist(ax_hp, res_p.paths, params_put, ACCENT2, ACCENT2, "Put")

    # ---- Convergence ----
    n_g, pr, ci_lo, ci_hi, bs = (study["n_grid"], study["prices"],
                                   study["ci_lower"], study["ci_upper"],
                                   study["bs_price"])
    ax_conv.fill_between(n_g, ci_lo, ci_hi, color=ACCENT4, alpha=0.25)
    ax_conv.plot(n_g, pr, color=ACCENT1, linewidth=1.6, label="MC estimate")
    ax_conv.axhline(bs, color=ACCENT3, linewidth=1.5, linestyle="--",
                    label=f"BS = {bs:.4f}")
    ax_conv.set_xscale("log")
    ax_conv.set_xlabel("N simulations", fontsize=10)
    ax_conv.set_ylabel("Call price", fontsize=10)
    ax_conv.set_title("Convergence vs N", fontsize=11)
    legend = ax_conv.legend(fontsize=9, framealpha=0, labelcolor=TEXT)
    _color_legend(legend)

    # super-title
    bs_c = BlackScholes.price(params_call)
    bs_p = BlackScholes.price(params_put)
    mc_c = pricer.price(params_call, n=n).price
    mc_p = pricer.price(params_put,  n=n).price
    fig.suptitle(
        f"Monte Carlo Option Pricing Dashboard  |  "
        f"S₀={params_call.S0}  K={params_call.K}  T={params_call.T}y  "
        f"r={params_call.r:.1%}  σ={params_call.sigma:.1%}\n"
        f"Call:  BS={bs_c:.4f}  MC={mc_c:.4f}    "
        f"Put:  BS={bs_p:.4f}  MC={mc_p:.4f}",
        color=TEXT, fontsize=11,
    )
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payoff_hist(ax, ST, params, color_otm, color_itm, label):
    K = params.K
    if params.option_type == "call":
        itm = ST >= K
    else:
        itm = ST <= K

    ax.hist(ST[~itm], bins=80, density=True, color=color_otm, alpha=0.4,
            edgecolor="none", label="OTM")
    ax.hist(ST[itm],  bins=80, density=True, color=color_itm, alpha=0.75,
            edgecolor="none", label="ITM")
    ax.axvline(K, color=ACCENT4, linestyle="--", linewidth=1.4)

    bs_price = BlackScholes.price(params)
    mc_price = np.exp(-params.r * params.T) * np.mean(
        np.maximum(ST - K, 0) if params.option_type == "call"
        else np.maximum(K - ST, 0)
    )
    ax.set_title(
        f"European {label}  |  BS={bs_price:.3f}  MC={mc_price:.3f}", fontsize=10
    )
    ax.set_xlabel("S_T", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    legend = ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT)
    _color_legend(legend)


def _color_legend(legend):
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_color(TEXT)
    legend.get_frame().set_facecolor(PANEL_BG)
    legend.get_frame().set_edgecolor(GRID)