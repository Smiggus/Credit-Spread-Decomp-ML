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
from src.panel import feature_columns


def model_inputs(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """The model's inputs: true/false columns as 0/1 (missing stays missing), text as categories."""
    X = df[columns or feature_columns()].copy()
    for col in X.columns:
        if pd.api.types.infer_dtype(X[col], skipna=True) == "boolean":
            X[col] = X[col].astype(float)
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category")
    return X


def fit_spread_model(train: pd.DataFrame, config: dict[str, Any] | None = None) -> xgb.Booster:
    """Fit XGBoost on the raw spread. Raises ValueError if any row is from the test months."""
    if (train["partition"] == "test").any():
        raise ValueError("Training rows include the test months, which stay closed until the pricer.")
    config = config or load_config()
    data = xgb.DMatrix(model_inputs(train), label=train[config["panel"]["target"]], enable_categorical=True)
    return xgb.train(config["spread_model"]["params"], data, num_boost_round=config["spread_model"]["rounds"])


def predict_spread(model: xgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """Predicted spread in bp, using the inputs the model was trained on."""
    return model.predict(xgb.DMatrix(model_inputs(df, model.feature_names), enable_categorical=True))
