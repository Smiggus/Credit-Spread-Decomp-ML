"""Acceptance tests for the company data in stage c (docs/specs/c_default.md), written before the code.

Each test is skipped until its issue is done; the PR that closes the issue removes the skip.
Imports sit inside the tests so the module loads before src/company.py exists.
All companies, codes and identifiers here are made up.
"""

import math

import pandas as pd
import pytest

T = pd.Timestamp


def months(start: str, end: str) -> pd.PeriodIndex:
    """Calendar months from start to end, inclusive."""
    return pd.period_range(start, end, freq="M")


DELISTINGS = pd.DataFrame({  # CRSP, already mapped to Compustat companies by the match
    "gvkey": ["001", "002"],
    "date": [T("2010-03-15"), T("2011-06-30")],
    "reason": ["performance", "merger"],
})
DELETIONS = pd.DataFrame({  # comp.company
    "gvkey": ["001", "003"],
    "dldte": [T("2010-07-20"), T("2012-01-10")],
    "dlrsn": ["02", "01"],
})


@pytest.mark.skip(reason="waits for #16")
def test_the_earlier_source_dates_the_failure():
    from src.company import failures

    out = failures(DELISTINGS, DELETIONS, codes=["performance"])
    assert out.set_index("gvkey")["date"].to_dict() == {"001": T("2010-03-15")}


@pytest.mark.skip(reason="waits for #16")
def test_a_merger_is_not_a_failure():
    from src.company import failures

    out = failures(DELISTINGS, DELETIONS, codes=["performance"])
    assert not {"002", "003"} & set(out["gvkey"])


def company_months() -> pd.DataFrame:
    """001 fails in March 2010; 002 leaves for a merger after December 2009."""
    a = pd.DataFrame({"gvkey": "001", "month": months("2009-01", "2010-06")})
    b = pd.DataFrame({"gvkey": "002", "month": months("2009-01", "2009-12")})
    return pd.concat([a, b], ignore_index=True)


@pytest.mark.skip(reason="waits for #16")
def test_a_month_is_labelled_by_a_failure_in_the_next_12_months():
    from src.company import labels

    failed = pd.DataFrame({"gvkey": ["001"], "date": [T("2010-03-15")]})
    out = labels(company_months(), failed, horizon=12).set_index(["gvkey", "month"])["label"]
    a = out.loc["001"]
    assert a.index.max() == pd.Period("2010-02", "M")  # from the failure month on, dropped
    assert a.loc[pd.Period("2009-02", "M")] == 0  # failure is 13 months ahead
    assert (a.loc[pd.Period("2009-03", "M"):] == 1).all()


@pytest.mark.skip(reason="waits for #16")
def test_a_company_that_leaves_for_a_merger_keeps_label_zero():
    from src.company import labels

    failed = pd.DataFrame({"gvkey": ["001"], "date": [T("2010-03-15")]})
    b = labels(company_months(), failed, horizon=12).query("gvkey == '002'")
    assert len(b) == 12 and (b["label"] == 0).all()


@pytest.mark.skip(reason="waits for #16")
def test_the_label_is_empty_when_the_window_runs_past_the_last_month():
    from src.company import labels

    no_failures = pd.DataFrame({"gvkey": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")})
    out = labels(company_months(), no_failures, horizon=12, last_month=pd.Period("2010-06", "M"))
    a = out.query("gvkey == '001'").set_index("month")["label"]
    assert a.loc[:pd.Period("2009-06", "M")].eq(0).all()  # window ends by June 2010
    assert a.loc[pd.Period("2009-07", "M"):].isna().all()


NAMES = pd.DataFrame({  # CRSP name history: one row per spell of a stock's CUSIP and ticker
    "permno": [10, 20, 30, 40, 60, 70, 70],
    "cusip": ["12345678", "12345678", "55555555", "66666666", "88888888", "11111111", "22222222"],
    "ticker": ["OLD", "NEW", "XYZ", "XYZ", "QRS", "CHA", "CHB"],
    "start": [T("2000-01-01"), T("2008-01-01"), T("2014-01-01"), T("2018-01-01"), T("2010-01-01"),
              T("2000-01-01"), T("2010-01-01")],
    "end": [T("2005-12-31"), T("2020-12-31"), T("2016-12-31"), T("2020-12-31"), T("2020-12-31"),
            T("2009-12-31"), T("2020-12-31")],
})
NO_VALUES = pd.DataFrame(columns=["permno", "month", "market_value"])


def compustat(rows: list[tuple]) -> pd.DataFrame:
    """Compustat company-months: gvkey, month, 9-digit CUSIP, ticker."""
    return pd.DataFrame(rows, columns=["gvkey", "month", "cusip", "ticker"]).assign(
        month=lambda d: pd.PeriodIndex(d["month"], freq="M"))


@pytest.mark.skip(reason="waits for #17")
def test_a_reused_cusip_matches_each_company_in_its_own_dates():
    from src.company import match_crsp

    rows = compustat([("A", "2003-06", "123456789", "OLD"), ("B", "2010-06", "123456789", "NEW")])
    out = match_crsp(rows, NAMES, NO_VALUES).set_index("gvkey")
    assert out["permno"].to_dict() == {"A": 10, "B": 20}
    assert set(out["method"]) == {"cusip"}


@pytest.mark.skip(reason="waits for #17")
def test_a_company_whose_cusip_changed_matches_through_its_current_cusip():
    from src.company import match_crsp

    out = match_crsp(compustat([("F", "2005-06", "222222229", "CHB")]), NAMES, NO_VALUES)
    assert (out["permno"].tolist(), out["method"].tolist()) == ([70], ["cusip"])


@pytest.mark.skip(reason="waits for #17")
def test_the_ticker_is_used_only_when_the_cusip_finds_nothing():
    from src.company import match_crsp

    rows = compustat([("C", "2015-06", "999999999", "XYZ"), ("D", "2015-06", "888888881", "XYZ")])
    out = match_crsp(rows, NAMES, NO_VALUES).set_index("gvkey")
    assert (out.loc["C", "permno"], out.loc["C", "method"]) == (30, "ticker")
    assert (out.loc["D", "permno"], out.loc["D", "method"]) == (60, "cusip")


@pytest.mark.skip(reason="waits for #17")
def test_with_two_stocks_the_larger_by_market_value_is_kept():
    from src.company import match_crsp

    names = pd.DataFrame({"permno": [50, 51], "cusip": ["77777777", "77777777"], "ticker": ["AA", "AB"],
                          "start": [T("2010-01-01")] * 2, "end": [T("2020-12-31")] * 2})
    values = pd.DataFrame({"permno": [50, 51], "month": pd.PeriodIndex(["2012-06", "2012-06"], freq="M"),
                           "market_value": [1e9, 4e9]})
    out = match_crsp(compustat([("E", "2012-06", "777777771", "AA")]), names, values)
    assert out["permno"].tolist() == [51]


@pytest.mark.skip(reason="waits for #18")
def test_nimta_average_weights_the_latest_quarter_most():
    from src.company import nimta_avg

    assert nimta_avg([0.04, 0.02, 0.0, -0.02]) == pytest.approx(0.025333, abs=1e-6)


@pytest.mark.skip(reason="waits for #18")
def test_accounting_data_is_used_from_its_report_date():
    from src.company import available_from

    quarters = pd.DataFrame({"datadate": [T("2010-03-31"), T("2010-06-30")], "rdq": [T("2010-04-25"), pd.NaT]})
    assert available_from(quarters).tolist() == [T("2010-04-25"), T("2010-09-28")]


@pytest.mark.skip(reason="waits for #18")
def test_price_is_capped_and_book_equity_adjusted():
    from src.company import adjusted_book, price_input

    assert price_input(20.0) == pytest.approx(math.log(15.0))
    assert price_input(10.0) == pytest.approx(math.log(10.0))
    assert adjusted_book(book=100.0, market=300.0) == pytest.approx(120.0)
