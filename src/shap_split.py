"""Stage a: split the predicted spread into the five input groups.

An XGBoost model on the raw spread, in basis points, so that its SHAP values add up in
basis points too. See docs/specs/a_shap_split.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import load_config
from src.panel import feature_columns, input_groups


def model_inputs(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """The model's inputs: true/false columns as 0/1 (missing stays missing), text as categories."""
    X = df[columns or feature_columns()].copy()
    for col in X.columns:
        if pd.api.types.infer_dtype(X[col], skipna=True) == "boolean":
            X[col] = X[col].astype(float)
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category")
    return X


def _refuse_test_rows(df: pd.DataFrame) -> None:
    if (df["partition"] == "test").any():
        raise ValueError("Rows include the test months, which stay closed until the pricer.")


def fit_spread_model(
    train: pd.DataFrame, columns: list[str] | None = None, config: dict[str, Any] | None = None
) -> xgb.Booster:
    """Fit XGBoost on the raw spread, on ``columns`` (default: all inputs in ``config``). Refuses test rows."""
    _refuse_test_rows(train)
    config = config or load_config()
    inputs = model_inputs(train, columns or feature_columns(config))
    data = xgb.DMatrix(inputs, label=train[config["panel"]["target"]], enable_categorical=True)
    return xgb.train(config["spread_model"]["params"], data, num_boost_round=config["spread_model"]["rounds"])


def _matrix(model: xgb.Booster, df: pd.DataFrame) -> xgb.DMatrix:
    """The rows of ``df`` as XGBoost input, with the columns the model was trained on."""
    return xgb.DMatrix(model_inputs(df, model.feature_names), enable_categorical=True)


def predict_spread(model: xgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """Predicted spread in bp."""
    return model.predict(_matrix(model, df))


def group_contributions(model: xgb.Booster, df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Each input group's SHAP contribution to the predicted spread, in bp, plus the ``base`` value.

    Row-aligned with ``df``. For every row, the group columns plus ``base`` sum to the prediction,
    up to XGBoost's float32 rounding (about 1e-6 of the spread).
    """
    if set(model.feature_names) != set(feature_columns(config)):
        raise ValueError("The model's inputs differ from the config's input groups: pass the config it was "
                         "fitted with.")
    raw = model.predict(_matrix(model, df), pred_contribs=True).astype(float)
    by_input = pd.DataFrame(raw[:, :-1], columns=model.feature_names, index=df.index)
    out = pd.DataFrame({name: by_input[cols].sum(axis=1) for name, cols in input_groups(config).items()})
    out["base"] = raw[:, -1]
    return out


def group_shares(contribs: pd.DataFrame, df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Each group's mean contribution (``mean_bp``) and share of the mean absolute contribution
    (``share_pct``) within each cut of ``df[by]``. One row per cut and group; no identifiers.

    ``contribs`` must be row-aligned with ``df`` (same index). A missing cut value is its own
    cut, so every cut covers every row.
    """
    if not contribs.index.equals(df.index):
        raise ValueError("contribs and df must share the same index (one row each, same order).")
    groups = contribs.drop(columns="base")
    keys = [df[col] for col in by]
    absolute = groups.abs().groupby(keys, dropna=False, observed=True).mean()
    shares = absolute.div(absolute.sum(axis=1), axis=0) * 100
    signed = groups.groupby(keys, dropna=False, observed=True).mean()

    def long(wide: pd.DataFrame, name: str) -> pd.DataFrame:
        return wide.reset_index().melt(id_vars=by, var_name="group", value_name=name)

    return long(signed, "mean_bp").merge(long(shares, "share_pct"), on=by + ["group"])


def drop_group_mae(train: pd.DataFrame, valid: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Valid MAE in bp by stratum for the full model (``left_out`` "none") and for the model
    refit without each input group in turn. Refuses test rows in either frame.
    """
    _refuse_test_rows(valid)
    config = config or load_config()
    stratum, target = config["panel"]["stratum"], config["panel"]["target"]
    groups = input_groups(config)
    rows = []
    for left_out in ["none"] + list(groups):
        columns = [col for col in feature_columns(config) if col not in groups.get(left_out, [])]
        error = (predict_spread(fit_spread_model(train, columns, config), valid) - valid[target]).abs()
        for name, mae in error.groupby(valid[stratum], dropna=False).mean().items():
            rows.append({"left_out": left_out, stratum: name, "mae_bp": mae})
    return pd.DataFrame(rows)
