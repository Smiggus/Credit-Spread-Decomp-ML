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

    table = monthly_measures(spark.createDataFrame(rows), months or T.SNAPSHOTS).toPandas()
    return table.set_index("cusip_id").to_dict("index")


@pytest.mark.skip(reason="waits for #9")
def test_export_refuses_a_local_path_inside_the_repo():
    from src.config import REPO_ROOT
    from src.trace import destination

    with pytest.raises(ValueError, match="inside the repo"):
        destination(str(REPO_ROOT / "data" / "trace"))
    assert destination("s3a://trace/enhanced") == "s3a://trace/enhanced"


@pytest.mark.skip(reason="waits for #10")
def test_cancelled_trade_and_its_original_are_removed(spark):
    from src.trace import clean_trades

    kept = clean_trades(spark.createDataFrame(T.CANCELLED)).toPandas()
    assert kept["msg_seq"].tolist() == [12]


@pytest.mark.skip(reason="waits for #10")
def test_correction_replaces_the_original(spark):
    from src.trace import clean_trades

    kept = clean_trades(spark.createDataFrame(T.CORRECTED)).toPandas()
    assert kept["rptd_pr"].tolist() == [100.5]


@pytest.mark.skip(reason="waits for #10")
def test_interdealer_trade_is_counted_once(spark):
    from src.trace import clean_trades

    assert clean_trades(spark.createDataFrame(T.INTERDEALER)).count() == 1


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
