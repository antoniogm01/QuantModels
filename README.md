# Monte Carlo Option Pricing under Black-Scholes

> A rigorous implementation of risk-neutral Monte Carlo simulation for pricing European options, with convergence analysis and comparison against the analytical Black-Scholes formula.

---

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Project Structure](#project-structure)
3. [Quickstart](#quickstart)
4. [Results](#results)
5. [Implementation Details](#implementation-details)
6. [Tests](#tests)
7. [References](#references)

---

## Theoretical Background

### The Black-Scholes SDE

Under the **risk-neutral measure** Q the stock price S_t obeys:

```
dS_t = r · S_t · dt  +  σ · S_t · dW_t^Q
```

where:
- `r` — continuously compounded risk-free rate  
- `σ` — constant annualised volatility  
- `W_t^Q` — standard Brownian motion under Q

### Closed-form Solution (Geometric Brownian Motion)

Applying **Itô's lemma** to `f(S_t) = ln S_t` yields:

```
d(ln S_t) = (r - σ²/2) dt  +  σ dW_t
```

Integrating over `[0, T]`:

```
S_T = S_0 · exp[ (r - σ²/2)·T  +  σ·√T·Z ],    Z ~ N(0,1)
```

This is the **exact simulation step** used in the Monte Carlo engine — no Euler-Maruyama discretisation error.

### European Option Pricing (Risk-Neutral Expectation)

By the **Fundamental Theorem of Asset Pricing**, the no-arbitrage price of a European derivative with payoff `h(S_T)` is:

```
V_0 = e^{-rT} · E^Q[ h(S_T) ]
```

For call and put options:

```
C = e^{-rT} · E^Q[ max(S_T − K, 0) ]
P = e^{-rT} · E^Q[ max(K − S_T, 0) ]
```

### Analytical Black-Scholes Formula (Benchmark)

```
C = S_0 · N(d₁)  −  K · e^{-rT} · N(d₂)
P = K · e^{-rT} · N(−d₂)  −  S₀ · N(−d₁)

d₁ = [ ln(S₀/K) + (r + σ²/2)·T ] / (σ·√T)
d₂ = d₁ − σ·√T
```

### Monte Carlo Estimator and Variance Reduction

The unbiased MC estimator for `N` paths is:

```
Ĉ = e^{-rT} · (1/N) · Σ max(S_T^(i) − K, 0)
```

Standard error scales as `O(N^{-1/2})`, confirmed empirically in the convergence study.

**Antithetic Variates**: for each draw `Z ~ N(0,1)` we also evaluate the payoff at `−Z`. The two are negatively correlated, cutting variance by ≈ 30–50% at zero extra cost.

### Put-Call Parity Verification

```
C − P = S₀ − K · e^{-rT}
```

The engine verifies this identity holds to machine precision (≈ 10⁻¹⁰).

---

## Project Structure

```
monte_carlo_options/
├── src/
│   ├── pricing_engine.py   # BlackScholes, MonteCarloPricer, OptionParams
│   └── visualisation.py    # All matplotlib figure generators
├── tests/
│   └── test_pricing.py     # pytest unit tests (15+ assertions)
├── outputs/                # Generated figures (git-ignored)
├── main.py                 # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/monte-carlo-options.git
cd monte-carlo-options

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# 4. Run tests
pytest tests/ -v
```

---

## Results

### Pricing Summary (S₀=100, K=100, T=1y, r=5%, σ=20%)

| Metric            | European Call | European Put |
|-------------------|:-------------:|:------------:|
| Black-Scholes     | 10.4506       | 5.5735       |
| Monte Carlo (N=1M)| 10.452 ± 0.014| 5.574 ± 0.011|
| Error (abs)       | < 0.005       | < 0.005      |
| Put-Call Parity   | ✓ (< 10⁻¹⁰)  |              |

### Convergence

The absolute error decays as **O(N^{-½})** as expected, confirmed on a log-log plot. At N=100,000 (with antithetic variates) the 95% confidence interval width is ≈ 0.06 — tighter than a typical bid-ask spread.

### Generated Figures

| File | Description |
|------|-------------|
| `1_gbm_paths.png` | 40 simulated GBM paths coloured by moneyness |
| `2_terminal_distribution.png` | Log-normal terminal price distribution vs analytical PDF |
| `3_convergence.png` | Price convergence and log-log error vs N |
| `4_price_surface.png` | BS call price heat-map over (spot, volatility) |
| `5_dashboard.png` | 4-panel summary dashboard |

---

## Implementation Details

### `OptionParams` (dataclass)

Immutable, validated contract specification — prevents silent bugs from mis-ordered arguments.

### `BlackScholes` (static class)

- `.price(params)` — closed-form call or put price  
- `.greeks(params)` — Delta, Gamma, Theta (per day), Vega (per vol point), Rho (per bp)

### `MonteCarloPricer`

- `.price(params, n, antithetic)` — returns `MCResult` with price, stderr, and 95% CI  
- `.simulate_full_paths(...)` — returns shape `(n_paths, n_steps+1)` for visualisation  
- `.convergence_study(...)` — sweeps N on a log grid and records all statistics

---

## Tests

```bash
pytest tests/ -v
```

Coverage includes:
- ATM call price against known textbook value  
- Put-call parity (< 10⁻⁸ error)  
- Deep ITM call approaches intrinsic value  
- All Greek signs (Δ, Γ, θ, ν, ρ)  
- MC price within tolerance of BS for both call and put  
- 95% CI contains the BS price  
- Standard error decreases with N  
- Antithetic variates reduce variance  

---

## References

1. Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy.
2. Merton, R. C. (1973). *Theory of Rational Option Pricing*. Bell Journal of Economics.
3. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
4. Hull, J. C. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.

---

## Licence

MIT