"""Acceptance tests for stage b (docs/specs/b_liquidity.md), written before the code.

Each test is skipped until its issue is done; the PR that closes the issue removes the skip.
Imports sit inside the tests so the module loads before src/trace.py and src/liquidity.py exist.
The tests run on Spark: the cluster when SPARK_REMOTE is set, else a local Spark.
"""

from datetime import date

import pandas as pd
import pytest

from tests import trades as T
from tests.synthetic import synthetic_panel


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


def kept_messages(spark, rows):
    """Message numbers of the trades that survive cleaning, in order."""
    from src.trace import clean_trades

    return sorted(int(m) for m in clean_trades(T.reports(spark, rows)).toPandas()["msg_seq_nb"])


def test_a_cancel_reported_days_later_removes_its_trade(spark):
    assert kept_messages(spark, T.CANCELLED_LATER) == [11, 12]


def test_a_correction_replaces_the_old_version(spark):
    from src.trace import clean_trades

    kept = clean_trades(T.reports(spark, T.CORRECTED)).toPandas()
    assert kept["rptd_pr"].tolist() == [100.5]


def test_a_reversal_removes_only_the_trade_it_points_to(spark):
    assert kept_messages(spark, T.REVERSED_TWIN) == [41]


def test_interdealer_buy_dropped_only_when_a_matching_dealer_sell_exists(spark):
    assert kept_messages(spark, T.INTERDEALER) == [30, 32, 34]


def test_special_and_late_settling_trades_are_dropped(spark):
    assert kept_messages(spark, T.SPECIAL) == [60, 66]


def test_reports_from_before_2012_are_refused(spark):
    from src.trace import clean_trades

    with pytest.raises(ValueError, match="2012"):
        clean_trades(T.reports(spark, T.OLD_PLATFORM))


def test_cleaning_counts_each_step(spark):
    from src.trace import cleaning_counts

    rows = T.CANCELLED_LATER + T.CORRECTED + T.REVERSED_TWIN + T.INTERDEALER + T.SPECIAL
    counts = cleaning_counts(T.reports(spark, rows))
    assert counts == {"start": 23, "not_a_trade_report": 4, "cancelled": 2, "reversed": 1,
                      "interdealer": 2, "special_or_late": 5, "end": 9}


def test_plain_trades_renames_for_the_measures(spark):
    from src.trace import clean_trades, plain_trades

    plain = plain_trades(clean_trades(T.reports(spark, T.SPECIAL)))
    assert plain.columns == ["cusip_id", "trd_exctn_dt", "trd_seq", "msg_seq", "rptd_pr", "par_usd"]


def test_amihud_on_a_hand_computed_day(spark):
    assert measures_of(spark, T.AMIHUD_DAY)["B1"]["amihud"] == pytest.approx(T.AMIHUD_EXPECTED)


def test_irc_of_two_same_size_trades(spark):
    assert measures_of(spark, T.IRC_DAY)["B1"]["irc"] == pytest.approx(T.IRC_EXPECTED)


def test_roll_on_bouncing_and_trending_prices(spark):
    assert measures_of(spark, T.ROLL_BOUNCE)["B1"]["roll"] == pytest.approx(T.ROLL_EXPECTED)
    assert measures_of(spark, T.ROLL_TREND)["B1"]["roll"] == 0


def test_zero_trade_share(spark):
    by_bond = measures_of(spark, T.ZERO_TRADE)
    assert by_bond["B2"]["zero_trade_share"] == pytest.approx(0.5)
    assert by_bond["B1"]["zero_trade_share"] == 0


def test_snapshot_day_trades_are_left_out(spark):
    # ROLL_BOUNCE trades once a day, so B1's only same-size pair on one day is on the snapshot day.
    by_bond = measures_of(spark, T.ON_SNAPSHOT_DAY + T.ROLL_BOUNCE)
    assert pd.isna(by_bond["B1"]["irc"])
    assert by_bond["B1"]["roll"] == pytest.approx(T.ROLL_EXPECTED)  # the snapshot-day prices are not in Roll


def lambda_rows(valid_value=2.0):
    """Three bonds whose four Amihud and IRC measures all equal one value; two train rows, one valid."""
    values = [("B0", "2025-01", 1.0), ("B1", "2025-01", 3.0), ("B2", "2025-08", valid_value)]
    return [{"cusip_id": b, "month": m, **{c: v for c in ("amihud", "amihud_risk", "irc", "irc_risk")}}
            for b, m, v in values]


def test_lambda_on_hand_computed_numbers(spark):
    from src.liquidity import add_lambda

    by_bond = add_lambda(spark.createDataFrame(lambda_rows()), ["2025-01"]).toPandas().set_index("cusip_id")
    # Train values 1 and 3: mean 2, sample sd sqrt(2). Each of the four measures scores -1/sqrt(2) for B0.
    assert by_bond.loc["B0", "lambda"] == pytest.approx(-4 / 2 ** 0.5)
    assert by_bond.loc["B2", "lambda"] == pytest.approx(0.0)


def test_lambda_scale_comes_from_train_months_only(spark):
    from src.liquidity import add_lambda

    base = add_lambda(spark.createDataFrame(lambda_rows()), ["2025-01"]).toPandas()
    moved = add_lambda(spark.createDataFrame(lambda_rows(valid_value=100.0)), ["2025-01"]).toPandas()
    train = base["month"] == "2025-01"
    assert base.loc[train, "lambda"].tolist() == moved.loc[train, "lambda"].tolist()


def test_lambda_refuses_a_measure_with_no_train_spread(spark):
    from src.liquidity import add_lambda

    rows = [dict(r, amihud=1.0) for r in lambda_rows()]
    with pytest.raises(ValueError, match="amihud"):
        add_lambda(spark.createDataFrame(rows), ["2025-01"])


def test_amihud_is_the_median_of_daily_values(spark):
    by_bond = measures_of(spark, T.AMIHUD_DAYS)
    assert by_bond["B1"]["amihud"] == pytest.approx(T.AMIHUD_MEDIAN)


def test_trades_in_the_same_second_are_ordered_by_message_number(spark):
    assert measures_of(spark, T.AMIHUD_TIED)["B1"]["amihud"] == pytest.approx(T.AMIHUD_TIED_EXPECTED)


def test_four_trades_of_one_size_are_not_a_roundtrip(spark):
    assert measures_of(spark, T.IRC_FOUR_AND_PAIR)["B1"]["irc"] == pytest.approx(0.01)


def test_roll_needs_three_price_changes(spark):
    assert pd.isna(measures_of(spark, T.ROLL_SHORT)["B1"]["roll"])


def test_zero_price_or_size_does_not_stop_the_run(spark):
    assert measures_of(spark, T.ZERO_PRICE)["B1"]["irc"] == pytest.approx(T.IRC_EXPECTED)


def measures_table(rows):
    """Monthly measures as the cluster brings them back: one row per bond and month."""
    return pd.DataFrame(rows, columns=["cusip_id", "month", "zero_trade_share", "amihud"])


def test_join_keeps_every_panel_row_and_its_index():
    from src.liquidity import join_measures

    panel = synthetic_panel(24).set_index(pd.RangeIndex(100, 124))
    first = panel.iloc[0]
    joined = join_measures(panel, measures_table([(first["cusip_id"], "2024-01", 0.25, 0.5)]))
    assert joined.index.equals(panel.index)
    assert list(joined.columns) == list(panel.columns) + ["zero_trade_share", "amihud"]
    assert joined["amihud"].notna().sum() == 1
    assert joined.iloc[0]["zero_trade_share"] == 0.25


def test_a_bond_with_no_trades_gets_zero_trade_share_one_in_covered_months_only():
    from src.liquidity import join_measures

    panel = synthetic_panel(24)  # one row per month from January 2024; row 0 is bond C0
    joined = join_measures(panel, measures_table([("C9", "2024-01", 0.5, 0.1)]))  # C0 did not trade
    assert joined.iloc[0]["zero_trade_share"] == 1.0
    assert pd.isna(joined.iloc[0]["amihud"])
    assert joined.iloc[1:]["zero_trade_share"].isna().all()  # months the measures do not cover


def test_join_refuses_two_measures_rows_for_one_bond_and_month():
    from src.liquidity import join_measures

    with pytest.raises(ValueError):
        join_measures(synthetic_panel(4), measures_table([("C0", "2024-01", 0.5, 0.1)] * 2))


def test_snapshot_days_gives_each_month_its_day():
    from src.liquidity import snapshot_days

    assert snapshot_days(synthetic_panel(2)) == {"2024-01": date(2024, 1, 31), "2024-02": date(2024, 2, 29)}


def test_snapshot_days_refuses_two_days_in_one_month():
    from src.liquidity import snapshot_days

    panel = synthetic_panel(2)
    panel.loc[1, "snapshot_date"] = pd.Timestamp("2024-01-15")
    with pytest.raises(ValueError, match="2024-01"):
        snapshot_days(panel)


def test_measures_join_the_liquidity_group_in_a_copy_of_the_config():
    from src.config import load_config
    from src.liquidity import with_measures

    config = load_config()
    groups = with_measures(config)["panel"]["input_groups"]
    assert groups["liquidity"] == config["panel"]["input_groups"]["liquidity"] + config["liquidity"]["measures"]
    assert "amihud" not in config["panel"]["input_groups"]["liquidity"]


def test_summary_counts_bonds_with_a_value_and_gives_quartiles():
    from src.liquidity import measure_summary

    joined = pd.DataFrame({"snapshot_date": pd.to_datetime(["2024-01-31"] * 5), "grade_bucket": ["IG"] * 4 + ["HY"],
                           "amihud": [1.0, 2.0, 3.0, None, 7.0]})
    summary = measure_summary(joined, ["amihud"], "grade_bucket")
    assert list(summary.columns) == ["measure", "grade_bucket", "month", "bonds", "p25", "p50", "p75"]
    by_grade = summary.set_index("grade_bucket")
    assert by_grade.loc["IG", ["bonds", "p25", "p50", "p75"]].tolist() == [3, 1.5, 2.0, 2.5]
    assert by_grade.loc["HY", "bonds"] == 1
