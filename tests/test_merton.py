"""Acceptance tests for the Merton spreads in stage c (docs/specs/c_default.md), written before the code.

Each test is skipped until its issue is done; the PR that closes the issue removes the skip.
Imports sit inside the tests so the module loaded before src/merton.py existed.
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


def test_hulls_example_gives_the_textbook_assets_and_default_probability():
    from src.merton import solve_assets

    assets, asset_vol = solve_assets(**HULL)
    assert assets == pytest.approx(12.40, abs=0.005)
    assert asset_vol == pytest.approx(0.2123, abs=0.00005)
    d2 = (math.log(assets / 10.0) + 0.05 - asset_vol**2 / 2) / asset_vol
    assert cdf(-d2) == pytest.approx(0.127, abs=0.0005)


def test_spread_matches_the_value_of_the_debt():
    from src.merton import model_spread, solve_assets

    assets, asset_vol = solve_assets(**HULL)
    spread = model_spread(assets, asset_vol, debt=10.0, rate=0.05, maturity=1.0)
    debt_value = assets - 3.0
    assert spread == pytest.approx((-math.log(debt_value / 10.0) - 0.05) * 1e4, abs=0.01)
    assert spread == pytest.approx(123.7, abs=0.05)


def test_spread_rises_with_debt_and_with_asset_volatility():
    from src.merton import model_spread

    base = model_spread(12.0, 0.2, 10.0, 0.05, 5.0)
    assert model_spread(12.0, 0.2, 11.0, 0.05, 5.0) > base
    assert model_spread(12.0, 0.3, 10.0, 0.05, 5.0) > base


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


def test_yields_in_percent_become_continuous_rates():
    from src.merton import continuous_rate

    assert continuous_rate(5.0) == pytest.approx(2 * math.log(1.025))
    np.testing.assert_allclose(continuous_rate(pd.Series([0.0, 4.0])), [0.0, 2 * math.log(1.02)])


def test_a_maturity_of_zero_or_less_gives_an_empty_spread():
    from src.merton import model_spread

    assert math.isnan(model_spread(12.0, 0.2, 10.0, 0.05, 0.0))


def test_the_spread_is_computed_for_whole_columns():
    from src.merton import model_spread

    out = model_spread(np.array([12.0, 12.0]), np.array([0.2, 0.2]), np.array([10.0, 0.0]),
                       np.array([0.05, 0.05]), np.array([5.0, 5.0]))
    assert out[0] == pytest.approx(model_spread(12.0, 0.2, 10.0, 0.05, 5.0)) and out[1] == 0.0


def test_a_company_month_with_no_solution_is_counted(monkeypatch):
    from src import merton

    monkeypatch.setattr(merton, "solve_assets", lambda *a, **k: (math.nan, math.nan))
    out, counts = merton.panel_spreads(panel_rows())
    assert out["merton_spread_bps"].isna().all()
    assert (counts["missing_input"], counts["no_solution"]) == (1, 2)


def summary_rows() -> pd.DataFrame:
    """IG 1-3y has five companies, HY 1-3y only three; one IG row has no model spread."""
    ig = pd.DataFrame({"issuer_ticker": [f"I{i}" for i in range(6)], "grade_bucket": "IG",
                       "spread_bps": [100.0] * 6, "merton_spread_bps": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan]})
    hy = pd.DataFrame({"issuer_ticker": ["H0", "H1", "H2"], "grade_bucket": "HY",
                       "spread_bps": [400.0] * 3, "merton_spread_bps": [100.0] * 3})
    rows = pd.concat([ig, hy], ignore_index=True)
    return rows.assign(partition="valid", maturity_bucket="1-3y", cusip_id=[f"B{i}" for i in range(len(rows))])


def test_summary_gives_means_quartiles_and_the_default_share():
    from src.merton import merton_summary

    table = merton_summary(summary_rows(), {"merton": "merton_spread_bps"})
    ig = table[(table["cut"] == "grade") & (table["grade_bucket"] == "IG")].iloc[0]
    assert (ig["model"], ig["rows"], ig["companies"]) == ("merton", 5, 5)
    assert (ig["model_mean_bp"], ig["model_p50_bp"], ig["observed_mean_bp"]) == (30.0, 30.0, 100.0)
    assert ig["default_share"] == pytest.approx(0.3)


def test_summary_blanks_cells_below_five_companies_and_has_no_identifiers():
    from src.merton import merton_summary

    table = merton_summary(summary_rows(), {"merton": "merton_spread_bps"})
    hy = table[table["grade_bucket"] == "HY"]
    assert (hy["companies"] == 3).all() and hy["model_mean_bp"].isna().all() and hy["observed_mean_bp"].isna().all()
    assert not {"cusip_id", "issuer_ticker", "issuer_name"} & set(table.columns)


def messy_rows() -> pd.DataFrame:
    """panel_rows on a shuffled index, plus four problem bonds.

    A bond with no ticker, a no-debt company with no maturity, a company with zero volatility,
    and a bond whose maturity is zero.
    """
    extra = pd.DataFrame({
        "snapshot_date": pd.Timestamp("2024-03-28"),
        "issuer_ticker": [None, "NOD", "ZER", "AAA"],
        "cusip_id": ["NONE00001", "NOD000001", "ZER000001", "AAA000003"],
        "log_market_cap": np.log([3e9, 2e9, 2e9, 3e9]),
        "vol_252d": [0.8, 0.3, 0.0, 0.8],
        "total_debt": [50e9, 0.0, 1e9, 10e9],
        "ust_2y_pct": 5.0,
        "years_to_maturity": [2.0, np.nan, 2.0, 0.0],
        "ust_yield_at_maturity_pct": 5.0,
    })
    rows = pd.concat([panel_rows(), extra], ignore_index=True)
    return rows.set_axis([10, 3, 7, 1, 22, 15, 4])


def test_problem_bonds_are_left_empty_and_the_counts_add_up():
    from src.merton import panel_spreads

    rows = messy_rows()
    out, counts = panel_spreads(rows)
    assert out.index.equals(rows.index)
    spread = out["merton_spread_bps"]
    assert spread.loc[[10, 3]].notna().all()  # AAA's 1y and 5y bonds
    assert spread.loc[[7, 1, 22, 15, 4]].isna().all()
    assert (counts["missing_input"], counts["bad_maturity"], counts["priced"]) == (4, 1, 2)
    assert counts["rows"] == counts["priced"] + counts["missing_input"] + counts["bad_maturity"] + counts["no_solution"]


def test_bonds_with_no_ticker_are_not_priced_as_one_company():
    from src.merton import panel_spreads

    rows = messy_rows()
    second = rows.loc[[1]].assign(cusip_id="NONE00002", total_debt=1e9).set_axis([99])
    out, _ = panel_spreads(pd.concat([rows, second]))
    assert out.loc[[1, 99], "merton_spread_bps"].isna().all()


def test_a_solver_failure_or_a_large_residual_gives_an_empty_result(monkeypatch):
    from types import SimpleNamespace

    from src import merton

    for fake in (SimpleNamespace(success=False, x=np.zeros(2), fun=np.zeros(2)),
                 SimpleNamespace(success=True, x=np.zeros(2), fun=np.array([1e-3, 0.0]))):
        monkeypatch.setattr(merton, "root", lambda *a, fake=fake, **k: fake)
        assert all(math.isnan(v) for v in merton.solve_assets(**HULL))


def test_a_tiny_spread_is_not_rounding_noise():
    from src.merton import model_spread

    # Assets of 100 times the debt at 20% volatility: the 1-year spread is about 1e-91 bp, never below 0.
    assert 0.0 <= model_spread(1000.0, 0.2, 10.0, 0.05, 1.0) < 1e-6
    assert model_spread(13.0, 0.25, 10.0, 0.05, 1.0) > 0


def test_summary_quartiles_in_the_maturity_cut():
    from src.merton import merton_summary

    table = merton_summary(summary_rows(), {"merton": "merton_spread_bps"})
    ig = table[(table["cut"] == "grade_maturity") & (table["grade_bucket"] == "IG")].iloc[0]
    assert (ig["maturity_bucket"], ig["model_p25_bp"], ig["model_p75_bp"]) == ("1-3y", 20.0, 40.0)


def cells(spec: list[tuple[str, int, int]]) -> pd.DataFrame:
    """IG valid rows: per maturity bucket, the number of companies and the rows of the first company."""
    parts = []
    for bucket, companies, first_rows in spec:
        tickers = [f"{bucket}-{i}" for i in range(companies)] + [f"{bucket}-0"] * (first_rows - 1)
        parts.append(pd.DataFrame({"issuer_ticker": tickers, "maturity_bucket": bucket}))
    rows = pd.concat(parts, ignore_index=True)
    return rows.assign(grade_bucket="IG", partition="valid", spread_bps=100.0, merton_spread_bps=10.0,
                       cusip_id=[f"B{i}" for i in range(len(rows))])


def test_a_lone_blank_cell_gets_a_partner_so_it_cannot_be_worked_out():
    from src.merton import merton_summary

    table = merton_summary(cells([("1-3y", 7, 1), ("3-5y", 5, 1), ("5-7y", 3, 1)]), {"merton": "merton_spread_bps"})
    by_bucket = table[table["cut"] == "grade_maturity"].set_index("maturity_bucket")["model_mean_bp"]
    assert by_bucket.isna().to_dict() == {"1-3y": False, "3-5y": True, "5-7y": True}


def test_a_cell_one_company_dominates_is_blanked():
    from src.merton import merton_summary

    table = merton_summary(cells([("1-3y", 6, 1), ("3-5y", 5, 7), ("5-7y", 6, 1)]), {"merton": "merton_spread_bps"})
    by_bucket = table[table["cut"] == "grade_maturity"].set_index("maturity_bucket")["model_mean_bp"]
    assert by_bucket.isna().to_dict() == {"1-3y": True, "3-5y": True, "5-7y": False}
