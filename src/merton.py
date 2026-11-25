"""Merton (1974) spreads for the panel's bonds (stage c, #14). See docs/specs/c_default.md.

Each company-month's asset value and asset volatility are solved once from its equity value,
equity volatility and debt. Each bond then gets the model spread at its own maturity.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.stats import norm

COMPANY_MONTH = ["issuer_ticker", "snapshot_date"]
COMPANY_INPUTS = ["log_market_cap", "vol_252d", "total_debt", "ust_2y_pct"]
RATE_AT_MATURITY = "ust_yield_at_maturity_pct"
SUMMARY_CUTS = {"grade": ["grade_bucket"], "grade_maturity": ["grade_bucket", "maturity_bucket"]}
MIN_COMPANIES = 5  # summary cells with fewer companies are left empty
MAX_COMPANY_SHARE = 0.5  # so are cells where one company holds more than half the rows
MAX_RESIDUAL = 1e-6  # the solved equations, with equity scaled to 1, must hold to this
VALUE_COLUMNS = ["observed_mean_bp", "model_mean_bp", "model_p25_bp", "model_p50_bp", "model_p75_bp", "default_share"]


def continuous_rate(yield_pct):
    """A yield in percent with semiannual compounding, as a continuous rate: 2 ln(1 + y / 200)."""
    rate = 2 * np.log1p(np.asarray(yield_pct, dtype=float) / 200)
    return float(rate) if rate.ndim == 0 else rate


def _cdf(x: float) -> float:
    """The standard normal distribution function for one number (faster than scipy's for scalars)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _equations(log_x, equity_vol, debt, rate, horizon):
    """The two Merton equations with equity scaled to 1, in log asset value and log asset volatility."""
    assets, asset_vol = np.exp(log_x)
    root_t = math.sqrt(horizon)
    d1 = (math.log(assets / debt) + (rate + asset_vol**2 / 2) * horizon) / (asset_vol * root_t)
    d2 = d1 - asset_vol * root_t
    return [assets * _cdf(d1) - debt * math.exp(-rate * horizon) * _cdf(d2) - 1.0,
            _cdf(d1) * asset_vol * assets - equity_vol]


def solve_assets(equity, equity_vol, debt, rate, horizon=1.0) -> tuple[float, float]:
    """Asset value and asset volatility for one company-month.

    Empty (NaN) when an input is missing or out of range, or when the solver fails or leaves
    the equations unsolved. With no debt the assets are the equity and carry no default risk.
    """
    values = (equity, equity_vol, debt, rate)
    if any(math.isnan(v) for v in values) or equity <= 0 or equity_vol <= 0 or debt < 0:
        return math.nan, math.nan
    if debt == 0:
        return float(equity), 0.0
    scaled_debt = debt / equity
    start = np.log([1.0 + scaled_debt, equity_vol / (1.0 + scaled_debt)])
    solution = root(_equations, start, args=(equity_vol, scaled_debt, rate, horizon))
    if not solution.success or np.max(np.abs(solution.fun)) > MAX_RESIDUAL:
        return math.nan, math.nan
    assets, asset_vol = np.exp(solution.x)
    return float(assets * equity), float(asset_vol)


def model_spread(assets, asset_vol, debt, rate, maturity):
    """The Merton spread in bp at the given maturity, for scalars or whole columns.

    0 with no debt; empty when an input is missing, the volatility is not positive with debt,
    or the maturity is not positive. Written with log1p and upper tails, so a tiny spread at
    low leverage is not lost to rounding.
    """
    a, s, f, r, t = (np.asarray(x, dtype=float) for x in (assets, asset_vol, debt, rate, maturity))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = (np.log(a / f) + (r + s**2 / 2) * t) / (s * np.sqrt(t))
        d2 = d1 - s * np.sqrt(t)
        spread = -np.log1p(a / (f * np.exp(-r * t)) * norm.sf(d1) - norm.sf(d2)) / t * 1e4
    spread = np.where(f == 0, 0.0, spread)
    invalid = ~np.isfinite(r) | ~(t > 0) | ((f > 0) & ~(s > 0))
    spread = np.where(invalid, np.nan, spread)
    return float(spread) if spread.ndim == 0 else spread


def _company_assets(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per company-month with its inputs and its solved asset value and volatility.

    Rows with no ticker or date are left out, so they cannot be merged into one company.
    """
    keyed = panel.dropna(subset=COMPANY_MONTH)
    firms = keyed.drop_duplicates(COMPANY_MONTH)[COMPANY_MONTH + COMPANY_INPUTS].reset_index(drop=True)
    solved = [solve_assets(math.exp(row.log_market_cap), row.vol_252d, row.total_debt, continuous_rate(row.ust_2y_pct))
              for row in firms.itertuples(index=False)]
    firms[["assets", "asset_vol"]] = pd.DataFrame(solved, columns=["assets", "asset_vol"])
    out_of_range = (firms["vol_252d"] <= 0) | (firms["total_debt"] < 0)
    firms["missing_input"] = firms[COMPANY_INPUTS].isna().any(axis=1) | out_of_range
    return firms


def _outcome_counts(missing, maturity, debt, spread) -> dict[str, int]:
    """Bond-months by outcome; rows = priced + missing_input + bad_maturity + no_solution."""
    priced = ~np.isnan(spread)
    bad_maturity = ~missing & (maturity <= 0)
    return {
        "rows": len(spread),
        "priced": int(priced.sum()),
        "missing_input": int(missing.sum()),
        "bad_maturity": int(bad_maturity.sum()),
        "no_solution": int((~missing & ~bad_maturity & ~priced).sum()),
        "no_debt": int((priced & (debt == 0)).sum()),
    }


def panel_spreads(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """The panel with ``merton_spread_bps`` added, and the counts of bond-months by outcome.

    Each company-month is solved once, and its debt is the one its bonds are priced against.
    A bond is left empty when its company or its own rate or maturity is missing.
    """
    rows = panel[COMPANY_MONTH].merge(_company_assets(panel), on=COMPANY_MONTH, how="left")
    rate = continuous_rate(panel[RATE_AT_MATURITY])
    maturity = panel["years_to_maturity"].to_numpy(dtype=float)
    spread = model_spread(rows["assets"], rows["asset_vol"], rows["total_debt"], rate, maturity)
    bond_missing = panel[[RATE_AT_MATURITY, "years_to_maturity"]].isna().any(axis=1).to_numpy()
    missing = rows["missing_input"].fillna(True).to_numpy(dtype=bool) | bond_missing
    spread = np.where(missing, np.nan, spread)
    counts = _outcome_counts(missing, maturity, rows["total_debt"].to_numpy(dtype=float), spread)
    return panel.assign(merton_spread_bps=spread), counts


def _model_cells(frame: pd.DataFrame, column: str, keys: list[str]) -> pd.DataFrame:
    """Rows, companies, means and quartiles per cell, over the rows that have a model spread."""
    rows = frame[frame[column].notna()]
    grouped = rows.groupby(keys, dropna=False)
    cells = grouped.agg(rows=(column, "size"), companies=("issuer_ticker", "nunique"),
                        observed_mean_bp=("spread_bps", "mean"), model_mean_bp=(column, "mean"))
    quartiles = grouped[column].quantile([0.25, 0.5, 0.75]).unstack()
    cells[["model_p25_bp", "model_p50_bp", "model_p75_bp"]] = quartiles.to_numpy()
    cells["default_share"] = cells["model_mean_bp"] / cells["observed_mean_bp"]
    per_company = rows.groupby(keys + ["issuer_ticker"], dropna=False).size()
    cells["top_company_rows"] = per_company.groupby(level=list(range(len(keys))), dropna=False).max().to_numpy()
    return cells.reset_index()


def _partners(table: pd.DataFrame) -> list:
    """Maturity cells to blank as well, so a blank cell cannot be worked out from its grade's total.

    Within each model, partition and grade: when exactly one maturity cell is blank, the published
    cell with the fewest companies is blanked too; when the grade total is blank, all of them are.
    """
    partners = []
    totals = table[table["cut"] == "grade"].set_index(["model", "partition", "grade_bucket"])["blank"]
    groups = table[table["cut"] == "grade_maturity"].groupby(["model", "partition", "grade_bucket"])
    for key, cells in groups:
        if totals.get(key, False):
            partners += list(cells.index)
        elif cells["blank"].sum() == 1 and len(cells) > 1:
            partners.append(cells.loc[~cells["blank"], "companies"].idxmin())
    return partners


def merton_summary(frame: pd.DataFrame, models: dict[str, str]) -> pd.DataFrame:
    """Each model's spread against the observed spread, by partition, grade and maturity. Totals only.

    models: model name to its spread column. A cell keeps its counts and loses its values when it
    has fewer than MIN_COMPANIES companies, when one company holds more than MAX_COMPANY_SHARE of
    its rows, or when it is needed to protect such a cell (see ``_partners``).
    """
    tables = [_model_cells(frame, column, ["partition"] + by).assign(cut=cut, model=name)
              for name, column in models.items() for cut, by in SUMMARY_CUTS.items()]
    table = pd.concat(tables, ignore_index=True)
    small = table["companies"] < MIN_COMPANIES
    table["blank"] = small | (table["top_company_rows"] > MAX_COMPANY_SHARE * table["rows"])
    table.loc[_partners(table), "blank"] = True
    table.loc[table["blank"], VALUE_COLUMNS] = np.nan
    first = ["model", "cut", "partition", "grade_bucket", "maturity_bucket", "rows", "companies"]
    return table.reindex(columns=first + VALUE_COLUMNS)
