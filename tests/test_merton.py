"""Acceptance tests for the Merton spreads in stage c (docs/specs/c_default.md), written before the code.

Each test is skipped until its issue is done; the PR that closes the issue removes the skip.
Imports sit inside the tests so the module loads before src/merton.py exists.
"""

import math

import numpy as np
import pandas as pd
import pytest


def cdf(x: float) -> float:
    """The standard normal distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Hull's worked example of default probability from equity prices.
HULL = {"equity": 3.0, "equity_vol": 0.80, "debt": 10.0, "rate": 0.05}


@pytest.mark.skip(reason="waits for #14")
def test_hulls_example_gives_the_textbook_assets_and_default_probability():
    from src.merton import solve_assets

    assets, asset_vol = solve_assets(**HULL)
    assert assets == pytest.approx(12.40, abs=0.005)
    assert asset_vol == pytest.approx(0.2123, abs=0.00005)
    d2 = (math.log(assets / 10.0) + 0.05 - asset_vol**2 / 2) / asset_vol
    assert cdf(-d2) == pytest.approx(0.127, abs=0.0005)


@pytest.mark.skip(reason="waits for #14")
def test_spread_matches_the_value_of_the_debt():
    from src.merton import model_spread, solve_assets

    assets, asset_vol = solve_assets(**HULL)
    spread = model_spread(assets, asset_vol, debt=10.0, rate=0.05, maturity=1.0)
    debt_value = assets - 3.0
    assert spread == pytest.approx((-math.log(debt_value / 10.0) - 0.05) * 1e4, abs=0.01)
    assert spread == pytest.approx(123.7, abs=0.05)


@pytest.mark.skip(reason="waits for #14")
def test_spread_rises_with_debt_and_with_asset_volatility():
    from src.merton import model_spread

    base = model_spread(12.0, 0.2, 10.0, 0.05, 5.0)
    assert model_spread(12.0, 0.2, 11.0, 0.05, 5.0) > base
    assert model_spread(12.0, 0.3, 10.0, 0.05, 5.0) > base


@pytest.mark.skip(reason="waits for #14")
def test_no_debt_gives_zero_and_a_missing_input_gives_an_empty_spread():
    from src.merton import model_spread, solve_assets

    assets, asset_vol = solve_assets(equity=5.0, equity_vol=0.3, debt=0.0, rate=0.04)
    assert (assets, asset_vol) == (5.0, 0.0)
    assert model_spread(assets, asset_vol, 0.0, 0.04, 5.0) == 0.0
    assets, asset_vol = solve_assets(equity=5.0, equity_vol=float("nan"), debt=2.0, rate=0.04)
    assert math.isnan(assets) and math.isnan(asset_vol)
    assert math.isnan(model_spread(assets, asset_vol, 2.0, 0.04, 5.0))


def panel_rows() -> pd.DataFrame:
    """Two companies in one month: AAA has bonds at 1 and 5 years, BBB has no volatility."""
    return pd.DataFrame({
        "snapshot_date": pd.Timestamp("2024-03-28"),
        "issuer_ticker": ["AAA", "AAA", "BBB"],
        "cusip_id": ["AAA000001", "AAA000002", "BBB000001"],
        "log_market_cap": np.log([3e9, 3e9, 5e9]),
        "vol_252d": [0.8, 0.8, np.nan],
        "total_debt": [10e9, 10e9, 1e9],
        "ust_2y_pct": [5.0, 5.0, 5.0],
        "years_to_maturity": [1.0, 5.0, 3.0],
        "ust_yield_at_maturity_pct": [5.0, 4.5, 4.8],
    })


@pytest.mark.skip(reason="waits for #14")
def test_each_company_month_is_solved_once_and_each_bond_gets_its_maturity(monkeypatch):
    from src import merton

    calls = []
    solve = merton.solve_assets
    monkeypatch.setattr(merton, "solve_assets", lambda *a, **k: calls.append(1) or solve(*a, **k))
    out, counts = merton.panel_spreads(panel_rows())
    assert len(calls) == 2
    aaa = out[out["issuer_ticker"] == "AAA"]["merton_spread_bps"]
    assert aaa.notna().all() and aaa.iloc[0] != aaa.iloc[1]
    assert np.isnan(out.loc[out["issuer_ticker"] == "BBB", "merton_spread_bps"]).all()
    assert counts["missing_input"] == 1
    assert out.index.equals(panel_rows().index)


@pytest.mark.skip(reason="waits for #15")
def test_a_zero_sharpe_ratio_keeps_the_historical_default_rate():
    from src.merton import calibrated_spread, risk_neutral_pd

    assert risk_neutral_pd(0.01, sharpe=0.0, maturity=1.0) == pytest.approx(0.01)
    assert calibrated_spread(0.01, sharpe=0.0, maturity=1.0, recovery=0.4) == pytest.approx(60.18, abs=0.01)


@pytest.mark.skip(reason="waits for #15")
def test_calibrated_spread_on_hand_computed_numbers():
    from src.merton import calibrated_spread, risk_neutral_pd

    pd_p = cdf(-2.0)
    assert risk_neutral_pd(pd_p, sharpe=0.5, maturity=4.0) == pytest.approx(cdf(-1.0))
    assert calibrated_spread(pd_p, sharpe=0.5, maturity=4.0, recovery=0.4) == pytest.approx(250.08, abs=0.01)


@pytest.mark.skip(reason="waits for #15")
def test_default_rate_is_interpolated_and_refused_past_the_table():
    from src.merton import default_rate

    table = {"IG": {1: 0.001, 3: 0.005}, "HY": {1: 0.04, 3: 0.12}}
    assert default_rate(table, "HY", 2.0) == pytest.approx(0.08)
    assert default_rate(table, "IG", 0.5) == pytest.approx(0.0005)  # from zero at horizon 0
    with pytest.raises(ValueError, match="past"):
        default_rate(table, "IG", 4.0)
