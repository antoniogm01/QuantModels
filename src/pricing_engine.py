"""
pricing_engine.py
=================
Monte Carlo option pricing under the Black-Scholes framework.

Theoretical Background
----------------------
Under the risk-neutral measure Q, the stock price S_t follows
Geometric Brownian Motion (GBM):

    dS_t = r * S_t * dt + σ * S_t * dW_t

where W_t is a standard Brownian motion under Q, r is the risk-free
rate, and σ is the volatility. By Itô's lemma, the exact solution is:

    S_T = S_0 * exp((r - σ²/2) * T + σ * √T * Z),  Z ~ N(0,1)

European option prices are discounted expectations under Q:
    C = e^{-rT} * E^Q[max(S_T - K, 0)]
    P = e^{-rT} * E^Q[max(K - S_T, 0)]

Analytical Black-Scholes formulae (closed-form benchmark):
    C = S_0 * N(d1) - K * e^{-rT} * N(d2)
    P = K * e^{-rT} * N(-d2) - S_0 * N(-d1)

where:
    d1 = [ln(S_0/K) + (r + σ²/2) * T] / (σ * √T)
    d2 = d1 - σ * √T
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionParams:
    """Encapsulates all parameters defining a European option contract."""
    S0: float          # Spot price
    K: float           # Strike price
    T: float           # Time to expiry (years)
    r: float           # Risk-free interest rate (continuous compounding)
    sigma: float       # Annualised volatility
    option_type: Literal["call", "put"] = "call"

    def __post_init__(self):
        assert self.S0 > 0,     "Spot price must be positive"
        assert self.K > 0,      "Strike must be positive"
        assert self.T > 0,      "Time to expiry must be positive"
        assert self.sigma > 0,  "Volatility must be positive"


@dataclass
class MCResult:
    """Container for Monte Carlo pricing results."""
    price: float
    stderr: float
    ci_lower: float
    ci_upper: float
    n_simulations: int
    paths: np.ndarray | None = None   # shape (n_simulations,) terminal prices


# ---------------------------------------------------------------------------
# Analytical Black-Scholes
# ---------------------------------------------------------------------------

class BlackScholes:
    """Analytical Black-Scholes pricer — used as the ground truth benchmark."""

    @staticmethod
    def d1(params: OptionParams) -> float:
        p = params
        return (np.log(p.S0 / p.K) + (p.r + 0.5 * p.sigma**2) * p.T) / (
            p.sigma * np.sqrt(p.T)
        )

    @staticmethod
    def d2(params: OptionParams) -> float:
        p = params
        return BlackScholes.d1(p) - p.sigma * np.sqrt(p.T)

    @classmethod
    def price(cls, params: OptionParams) -> float:
        p = params
        d1, d2 = cls.d1(p), cls.d2(p)
        if p.option_type == "call":
            return p.S0 * norm.cdf(d1) - p.K * np.exp(-p.r * p.T) * norm.cdf(d2)
        else:
            return p.K * np.exp(-p.r * p.T) * norm.cdf(-d2) - p.S0 * norm.cdf(-d1)

    @classmethod
    def greeks(cls, params: OptionParams) -> dict:
        """Return the first-order Greeks for the option."""
        p = params
        d1, d2 = cls.d1(p), cls.d2(p)
        sqrt_T = np.sqrt(p.T)
        disc   = np.exp(-p.r * p.T)

        delta = norm.cdf(d1)  if p.option_type == "call" else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (p.S0 * p.sigma * sqrt_T)
        theta_call = (
            -p.S0 * norm.pdf(d1) * p.sigma / (2 * sqrt_T)
            - p.r * p.K * disc * norm.cdf(d2)
        )
        theta = theta_call if p.option_type == "call" else (
            theta_call + p.r * p.K * disc
        )
        vega  = p.S0 * norm.pdf(d1) * sqrt_T
        rho   = (p.K * p.T * disc * norm.cdf(d2)  if p.option_type == "call"
                 else -p.K * p.T * disc * norm.cdf(-d2))

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta / 365,   # per calendar day
            "vega":  vega / 100,    # per 1 vol-point
            "rho":   rho / 100,     # per 1 bp
        }


# ---------------------------------------------------------------------------
# Monte Carlo pricer
# ---------------------------------------------------------------------------

class MonteCarloPricer:
    """
    Monte Carlo pricer via exact GBM simulation.

    The terminal stock price is drawn directly from the risk-neutral
    log-normal distribution — no Euler discretisation error.

    Variance reduction
    ------------------
    Antithetic variates are used by default: for each standard normal
    draw Z we also evaluate the payoff at -Z, halving the effective
    variance at negligible extra cost.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # GBM simulation
    # ------------------------------------------------------------------

    def simulate_terminal_prices(
        self,
        params: OptionParams,
        n: int,
        antithetic: bool = True,
    ) -> np.ndarray:
        """
        Simulate n terminal stock prices under risk-neutral GBM.

        S_T = S_0 · exp((r - σ²/2)·T + σ·√T·Z),  Z ~ N(0,1)

        With antithetic variates the returned array has length 2n.
        """
        p = params
        drift    = (p.r - 0.5 * p.sigma**2) * p.T
        diffusion = p.sigma * np.sqrt(p.T)

        Z = self.rng.standard_normal(n)
        if antithetic:
            Z = np.concatenate([Z, -Z])

        return p.S0 * np.exp(drift + diffusion * Z)

    def simulate_full_paths(
        self,
        params: OptionParams,
        n_paths: int,
        n_steps: int = 252,
    ) -> np.ndarray:
        """
        Simulate full GBM paths (for visualisation purposes).

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
        """
        p  = params
        dt = p.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = p.S0

        Z = self.rng.standard_normal((n_paths, n_steps))
        increments = (p.r - 0.5 * p.sigma**2) * dt + p.sigma * np.sqrt(dt) * Z
        paths[:, 1:] = p.S0 * np.exp(np.cumsum(increments, axis=1))
        return paths

    # ------------------------------------------------------------------
    # Option pricing
    # ------------------------------------------------------------------

    def price(
        self,
        params: OptionParams,
        n: int = 100_000,
        antithetic: bool = True,
        store_paths: bool = False,
    ) -> MCResult:
        """
        Price a European option via Monte Carlo.

        Parameters
        ----------
        params      : OptionParams
        n           : number of simulations (paths doubled if antithetic)
        antithetic  : use antithetic variates for variance reduction
        store_paths : whether to retain terminal prices in the result

        Returns
        -------
        MCResult with price estimate, standard error, and 95% CI
        """
        ST = self.simulate_terminal_prices(params, n, antithetic)
        K  = params.K
        disc = np.exp(-params.r * params.T)

        if params.option_type == "call":
            payoffs = np.maximum(ST - K, 0.0)
        else:
            payoffs = np.maximum(K - ST, 0.0)

        discounted = disc * payoffs
        price  = discounted.mean()
        stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))

        return MCResult(
            price        = price,
            stderr       = stderr,
            ci_lower     = price - 1.96 * stderr,
            ci_upper     = price + 1.96 * stderr,
            n_simulations= len(discounted),
            paths        = ST if store_paths else None,
        )

    # ------------------------------------------------------------------
    # Convergence study
    # ------------------------------------------------------------------

    def convergence_study(
        self,
        params: OptionParams,
        n_grid: np.ndarray | None = None,
        antithetic: bool = True,
    ) -> dict:
        """
        Compute MC price at increasing simulation counts.

        Returns a dict with keys:
            n_grid, prices, stderr, ci_lower, ci_upper, bs_price
        """
        if n_grid is None:
            n_grid = np.logspace(2, 6, 40, dtype=int)
            n_grid = np.unique(n_grid)

        bs_price = BlackScholes.price(params)
        prices, stderrs, ci_lo, ci_hi = [], [], [], []

        for n in n_grid:
            res = self.price(params, n=int(n), antithetic=antithetic)
            prices.append(res.price)
            stderrs.append(res.stderr)
            ci_lo.append(res.ci_lower)
            ci_hi.append(res.ci_upper)

        return {
            "n_grid":   n_grid,
            "prices":   np.array(prices),
            "stderr":   np.array(stderrs),
            "ci_lower": np.array(ci_lo),
            "ci_upper": np.array(ci_hi),
            "bs_price": bs_price,
        }