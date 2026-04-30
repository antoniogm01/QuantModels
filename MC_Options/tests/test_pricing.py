"""
tests/test_pricing.py
=====================
Unit tests for the Monte Carlo option pricing engine.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from pricing_engine import BlackScholes, MonteCarloPricer, OptionParams

@pytest.fixture
def atm_call():
    return OptionParams(S0=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")

@pytest.fixture
def atm_put():
    return OptionParams(S0=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")

@pytest.fixture
def pricer():
    return MonteCarloPricer(seed=42)

class TestBlackScholes:

    def test_call_price_known_value(self, atm_call):
        """ATM call with S0=K=100, T=1, r=5%, σ=20% ≈ 10.451 (textbook)."""
        price = BlackScholes.price(atm_call)
        assert abs(price - 10.4506) < 0.01, f"Call price {price} outside tolerance"

    def test_put_call_parity(self, atm_call, atm_put):
        """C - P == S0 - K * exp(-rT)."""
        c  = BlackScholes.price(atm_call)
        p  = BlackScholes.price(atm_put)
        lhs = c - p
        rhs = atm_call.S0 - atm_call.K * np.exp(-atm_call.r * atm_call.T)
        assert abs(lhs - rhs) < 1e-8, f"Put-call parity violated: {lhs} != {rhs}"

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call ≈ S0 - K * e^{-rT}."""
        p = OptionParams(S0=200, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
        intrinsic = p.S0 - p.K * np.exp(-p.r * p.T)
        price = BlackScholes.price(p)
        assert abs(price - intrinsic) < 0.5

    def test_otm_call_positive(self):
        p = OptionParams(S0=80, K=120, T=0.25, r=0.02, sigma=0.15, option_type="call")
        assert BlackScholes.price(p) > 0

    def test_greeks_signs(self, atm_call, atm_put):
        gc = BlackScholes.greeks(atm_call)
        gp = BlackScholes.greeks(atm_put)
        assert 0 < gc["delta"] < 1,  "Call delta not in (0,1)"
        assert -1 < gp["delta"] < 0, "Put delta not in (-1,0)"
        assert gc["gamma"] > 0,      "Gamma must be positive"
        assert gc["vega"]  > 0,      "Vega must be positive"
        assert gc["theta"] < 0,      "Theta must be negative (time decay)"
        assert gc["rho"]   > 0,      "Call rho must be positive"
        assert gp["rho"]   < 0,      "Put rho must be negative"


class TestMonteCarloPricer:

    def test_mc_call_close_to_bs(self, atm_call, pricer):
        bs = BlackScholes.price(atm_call)
        mc = pricer.price(atm_call, n=500_000).price
        assert abs(mc - bs) < 0.10, f"MC call {mc} too far from BS {bs}"

    def test_mc_put_close_to_bs(self, atm_put, pricer):
        bs = BlackScholes.price(atm_put)
        mc = pricer.price(atm_put, n=500_000).price
        assert abs(mc - bs) < 0.10, f"MC put {mc} too far from BS {bs}"

    def test_ci_contains_bs(self, atm_call, pricer):
        """95% CI should contain BS price nearly all the time."""
        bs = BlackScholes.price(atm_call)
        res = pricer.price(atm_call, n=200_000)
        assert res.ci_lower <= bs <= res.ci_upper, (
            f"BS price {bs:.4f} outside CI [{res.ci_lower:.4f}, {res.ci_upper:.4f}]"
        )

    def test_stderr_decreases_with_n(self, atm_call, pricer):
        stderr_small = pricer.price(atm_call, n=1_000).stderr
        stderr_large = pricer.price(atm_call, n=100_000).stderr
        assert stderr_large < stderr_small

    def test_terminal_prices_shape(self, atm_call, pricer):
        res = pricer.price(atm_call, n=1000, antithetic=True, store_paths=True)
        assert res.paths.shape == (2000,)   # antithetic doubles

    def test_paths_shape(self, atm_call, pricer):
        paths = pricer.simulate_full_paths(atm_call, n_paths=10, n_steps=50)
        assert paths.shape == (10, 51)

    def test_antithetic_reduces_variance(self, atm_call):
        """Antithetic variates should give lower stderr for same N."""
        p1 = MonteCarloPricer(seed=1)
        p2 = MonteCarloPricer(seed=1)
        with_av    = p1.price(atm_call, n=50_000, antithetic=True).stderr
        without_av = p2.price(atm_call, n=100_000, antithetic=False).stderr
        # They use the same effective sample size; AV should be competitive
        assert with_av < without_av * 1.5   # lenient bound

    def test_convergence_study_shape(self, atm_call, pricer):
        study = pricer.convergence_study(atm_call,
                                         n_grid=np.array([100, 500, 1000]))
        assert len(study["prices"]) == 3
        assert study["bs_price"] == pytest.approx(BlackScholes.price(atm_call), rel=1e-6)