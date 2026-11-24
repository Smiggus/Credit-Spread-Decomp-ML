"""Acceptance tests for the three-part split in stage c (docs/specs/c_default.md), written before the code.

Each test is skipped until its issue is done; the PR that closes the issue removes the skip.
Imports sit inside the tests so the module loads before src/split.py exists.
"""

import copy

import numpy as np
import pytest

from src.config import load_config
from tests.synthetic import driven_panel

PARTS = ["default", "common", "liquidity", "market", "unexplained"]


def panel_with_default():
    """Train and valid rows of a synthetic panel driven by liquidity and market inputs.

    The default spread is a fifth of the spread.
    """
    df = driven_panel(["n_trades", "vix"], n=600)
    df = df[df["partition"].isin(["train", "valid"])]
    return df.assign(default_bps=0.2 * df["spread_bps"].clip(lower=0))


@pytest.mark.skip(reason="waits for #20")
def test_default_spread_is_the_annual_expected_loss():
    from src.split import default_spread

    assert default_spread(0.02, recovery=0.4) == pytest.approx(120.0)


@pytest.mark.skip(reason="waits for #20")
def test_the_parts_add_up_to_the_spread_for_every_bond_month():
    from src.split import split_parts

    df = panel_with_default()
    parts = split_parts(df, "default_bps", load_config())
    assert parts.index.equals(df.index)
    np.testing.assert_allclose(parts[PARTS].sum(axis=1), df["spread_bps"], atol=1e-6)


@pytest.mark.skip(reason="waits for #20")
def test_the_residual_model_refuses_inputs_outside_liquidity_and_market():
    from src.split import split_parts

    config = copy.deepcopy(load_config())
    config["split"]["residual_groups"] = ["liquidity", "market", "bond_terms"]
    with pytest.raises(ValueError, match="liquidity and market"):
        split_parts(panel_with_default(), "default_bps", config)


@pytest.mark.skip(reason="waits for #20")
def test_the_split_table_has_no_identifiers_and_covers_every_row():
    from src.split import split_parts, split_table

    df = panel_with_default()
    table = split_table(split_parts(df, "default_bps", load_config()), df)
    assert not {"cusip_id", "issuer_ticker", "issuer_name"} & set(table.columns)
    assert table["rows"].sum() == len(df)
    assert set(table["grade_bucket"]) == {"IG", "HY"}
