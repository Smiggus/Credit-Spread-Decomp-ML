"""Acceptance tests for stage b (docs/specs/b_liquidity.md), written before the code.

Each test is skipped until its issue is done; the PR that closes the issue removes the skip.
Imports sit inside the tests so the module loads before src/trace.py and src/liquidity.py exist.
The tests run on Spark: the cluster when SPARK_REMOTE is set, else a local Spark.
"""

import pandas as pd
import pytest

from tests import trades as T


@pytest.fixture(name="spark", scope="module")
def spark_session():
    """A Spark session from src.spark: the cluster if SPARK_REMOTE is set, else local."""
    from src.spark import get_spark

    return get_spark()


def measures_of(spark, rows, months=None):
    """Monthly measures for made-up trades, as a dict per bond."""
    from src.liquidity import monthly_measures

    table = monthly_measures(T.frame(spark, rows), months or T.SNAPSHOTS).toPandas()
    return table.set_index("cusip_id").to_dict("index")


REQUIRED = ["cusip_id", "rptd_pr"]


def manifest(month, rows, columns=("cusip_id", "rptd_pr", "trc_st")):
    """A month's manifest as the data platform writes it (only the fields the check reads)."""
    return {"table": "trace_enhanced", "month": month, "rows": rows, "columns": [{"name": c} for c in columns]}


def test_check_accepts_a_complete_delivery():
    from src.trace import check_delivery

    manifests = {m: manifest(m, 2) for m in ("2025-03", "2025-04")}
    check_delivery(manifests, {"2025-03": 2, "2025-04": 2}, ["2025-03", "2025-04"], REQUIRED)


def test_check_accepts_an_empty_month_with_no_files():
    from src.trace import check_delivery

    check_delivery({"2025-03": manifest("2025-03", 0)}, {}, ["2025-03"], REQUIRED)


def test_check_names_a_missing_month():
    from src.trace import check_delivery

    with pytest.raises(ValueError, match="2025-04"):
        check_delivery({"2025-03": manifest("2025-03", 2)}, {"2025-03": 2}, ["2025-03", "2025-04"], REQUIRED)


def test_check_names_a_month_whose_rows_differ_from_its_manifest():
    from src.trace import check_delivery

    with pytest.raises(ValueError, match="2025-03"):
        check_delivery({"2025-03": manifest("2025-03", 3)}, {"2025-03": 2}, ["2025-03"], REQUIRED)


def test_check_names_a_missing_column():
    from src.trace import check_delivery

    month = {"2025-03": manifest("2025-03", 2, columns=("cusip_id",))}
    with pytest.raises(ValueError, match="rptd_pr"):
        check_delivery(month, {"2025-03": 2}, ["2025-03"], REQUIRED)


def test_check_names_the_month_that_lacks_a_column():
    from src.trace import check_delivery

    manifests = {"2025-03": manifest("2025-03", 2), "2025-04": manifest("2025-04", 2, columns=("cusip_id",))}
    with pytest.raises(ValueError, match="2025-04"):
        check_delivery(manifests, {"2025-03": 2, "2025-04": 2}, ["2025-03", "2025-04"], REQUIRED)


def test_reader_refuses_a_local_path_inside_the_repo():
    from src.config import REPO_ROOT
    from src.trace import read_trades

    with pytest.raises(ValueError, match="inside the repo"):
        read_trades(None, str(REPO_ROOT / "data" / "trace"), "trace_enhanced", ["2025-03"])


def test_cancelled_trade_and_its_original_are_removed(spark):
    from src.trace import clean_trades

    kept = clean_trades(T.frame(spark, T.CANCELLED)).toPandas()
    assert kept["msg_seq"].tolist() == ["12"]


def test_correction_replaces_the_original(spark):
    from src.trace import clean_trades

    kept = clean_trades(T.frame(spark, T.CORRECTED)).toPandas()
    assert kept["rptd_pr"].tolist() == [100.5]


def test_interdealer_trade_is_counted_once(spark):
    from src.trace import clean_trades

    kept = clean_trades(T.frame(spark, T.INTERDEALER)).toPandas()
    assert kept["side"].tolist() == ["S"]


def test_reversal_and_the_trade_it_reverses_are_removed(spark):
    from src.trace import clean_trades

    kept = clean_trades(T.frame(spark, T.REVERSED)).toPandas()
    assert kept["msg_seq"].tolist() == ["42"]


def test_cleaning_counts_each_step(spark):
    from src.trace import cleaning_counts

    rows = T.CANCELLED + T.CORRECTED + T.INTERDEALER + T.REVERSED
    counts = cleaning_counts(T.frame(spark, rows))
    assert counts == {"start": 10, "cancelled": 2, "corrected": 1, "reversed": 2, "interdealer_buy_side": 1,
                      "unknown_status": 0, "end": 4}


def test_wrds_codes_map_to_plain_status(spark):
    from src.trace import normalise

    raw = spark.createDataFrame(T.WRDS_CODED, schema=T.WRDS_SCHEMA)
    plain = normalise(raw).toPandas()
    status = {seq.split(":")[1].split(".")[0]: s for seq, s in zip(plain["msg_seq"], plain["status"])}
    assert status == T.WRDS_STATUS
    assert plain["orig_msg_seq"].notna().sum() == 4  # only cancels and corrections point to an original


@pytest.mark.skip(reason="waits for #11")
def test_amihud_on_a_hand_computed_day(spark):
    assert measures_of(spark, T.AMIHUD_DAY)["B1"]["amihud"] == pytest.approx(T.AMIHUD_EXPECTED)


@pytest.mark.skip(reason="waits for #11")
def test_irc_of_two_same_size_trades(spark):
    assert measures_of(spark, T.IRC_DAY)["B1"]["irc"] == pytest.approx(T.IRC_EXPECTED)


@pytest.mark.skip(reason="waits for #11")
def test_roll_on_bouncing_and_trending_prices(spark):
    assert measures_of(spark, T.ROLL_BOUNCE)["B1"]["roll"] == pytest.approx(T.ROLL_EXPECTED)
    assert measures_of(spark, T.ROLL_TREND)["B1"]["roll"] == 0


@pytest.mark.skip(reason="waits for #11")
def test_zero_trade_share(spark):
    by_bond = measures_of(spark, T.ZERO_TRADE)
    assert by_bond["B2"]["zero_trade_share"] == pytest.approx(0.5)
    assert by_bond["B1"]["zero_trade_share"] == 0


@pytest.mark.skip(reason="waits for #11")
def test_snapshot_day_trades_are_left_out(spark):
    by_bond = measures_of(spark, T.ON_SNAPSHOT_DAY + T.AMIHUD_DAY)
    assert pd.isna(by_bond["B1"]["irc"])  # the only same-size pair is on the snapshot day


@pytest.mark.skip(reason="waits for #11")
def test_lambda_scale_comes_from_train_months_only(spark):
    from src.liquidity import add_lambda

    rows = [{"cusip_id": f"B{i}", "month": m, "amihud": a, "amihud_risk": a, "irc": a, "irc_risk": a}
            for i, (m, a) in enumerate([("2025-01", 1.0), ("2025-01", 3.0), ("2025-08", 2.0)])]
    changed = [dict(r, amihud=100.0) if r["month"] == "2025-08" else r for r in rows]
    base = add_lambda(spark.createDataFrame(rows), train_months=["2025-01"]).toPandas()
    moved = add_lambda(spark.createDataFrame(changed), train_months=["2025-01"]).toPandas()
    # Changing a valid month's value moves its own lambda but not the train months' lambdas.
    train = base["month"] == "2025-01"
    assert base.loc[train, "lambda"].tolist() == moved.loc[train, "lambda"].tolist()


@pytest.mark.skip(reason="waits for #12")
def test_join_keeps_every_panel_row():
    from src.liquidity import join_measures
    from tests.synthetic import synthetic_panel

    panel = synthetic_panel(24)
    measures = pd.DataFrame({"cusip_id": ["C0"], "month": [panel["snapshot_date"].iloc[0].strftime("%Y-%m")],
                             "amihud": [0.5]})
    joined = join_measures(panel, measures)
    assert len(joined) == len(panel)
    assert joined["amihud"].notna().sum() == 1
