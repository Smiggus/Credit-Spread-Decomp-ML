"""Acceptance tests for stage a (docs/specs/a_shap_split.md), written before the code.

Each test is skipped until its issue is done; the PR that closes the issue removes the skip.
Imports sit inside the tests so the module loads before src/shap_split.py exists.
"""

import numpy as np
import pytest

from src.panel import input_groups
from tests.synthetic import driven_panel, synthetic_panel

LIQUIDITY = ["n_trades", "log_par_volume", "log_par_volume_month"]


def split(df):
    """Train and valid rows of a synthetic panel."""
    return df[df["partition"] == "train"], df[df["partition"] == "valid"]


@pytest.mark.skip(reason="waits for #4")
def test_fit_refuses_test_rows():
    from src.shap_split import fit_spread_model

    df = synthetic_panel(60)
    with pytest.raises(ValueError, match="test"):
        fit_spread_model(df)


@pytest.mark.skip(reason="waits for #4")
def test_model_beats_the_train_mean_on_valid():
    from src.shap_split import fit_spread_model, predict_spread

    train, valid = split(driven_panel(LIQUIDITY))
    model = fit_spread_model(train)
    model_mae = np.abs(predict_spread(model, valid) - valid["spread_bps"]).mean()
    mean_mae = np.abs(train["spread_bps"].mean() - valid["spread_bps"]).mean()
    assert model_mae < mean_mae


@pytest.mark.skip(reason="waits for #5")
def test_contributions_add_up_to_the_prediction():
    from src.shap_split import fit_spread_model, group_contributions, predict_spread

    train, valid = split(driven_panel(LIQUIDITY))
    model = fit_spread_model(train)
    contribs = group_contributions(model, valid)
    assert list(contribs.columns) == list(input_groups()) + ["base"]
    np.testing.assert_allclose(contribs.sum(axis=1), predict_spread(model, valid), atol=1e-3)


@pytest.mark.skip(reason="waits for #5")
def test_liquidity_driven_spread_gives_liquidity_the_largest_share():
    from src.shap_split import fit_spread_model, group_contributions

    train, valid = split(driven_panel(LIQUIDITY))
    contribs = group_contributions(fit_spread_model(train), valid).drop(columns="base")
    shares = contribs.abs().mean() / contribs.abs().mean().sum()
    assert shares["liquidity"] > 0.8


@pytest.mark.skip(reason="waits for #6")
def test_shares_sum_to_100_within_each_cut():
    from src.shap_split import fit_spread_model, group_contributions, group_shares

    train, valid = split(driven_panel(LIQUIDITY))
    contribs = group_contributions(fit_spread_model(train), valid)
    table = group_shares(contribs, valid, by=["grade_bucket", "maturity_bucket"])
    totals = table.groupby(["grade_bucket", "maturity_bucket"])["share_pct"].sum()
    np.testing.assert_allclose(totals, 100, atol=1e-6)


@pytest.mark.skip(reason="waits for #6")
def test_shares_table_has_no_identifiers():
    from src.shap_split import fit_spread_model, group_contributions, group_shares

    train, valid = split(driven_panel(LIQUIDITY))
    contribs = group_contributions(fit_spread_model(train), valid)
    table = group_shares(contribs, valid, by=["snapshot_date"])
    assert not {"cusip_id", "issuer_ticker"} & set(table.columns)


@pytest.mark.skip(reason="waits for #7")
def test_leaving_out_the_driving_group_hurts_most():
    from src.shap_split import drop_group_mae

    train, valid = split(driven_panel(LIQUIDITY))
    table = drop_group_mae(train, valid)
    rise = table.groupby("left_out")["mae_bp"].mean()
    rise = rise.drop("none") - rise["none"]
    assert rise.idxmax() == "liquidity"
