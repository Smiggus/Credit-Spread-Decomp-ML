"""Acceptance tests for the default classifier ladder in stage c (docs/specs/c_default.md), written before the code.

Each test is skipped until its issue is done; the PR that closes the issue removes the skip.
Imports sit inside the tests so the module loads before src/default_models.py exists.
"""

import numpy as np
import pandas as pd
import pytest

INPUTS = ["x0", "x1", "x2"]


def planted(n: int = 3000, seed: int = 0, start: str = "1995-01", end: str = "2012-12") -> pd.DataFrame:
    """Company-months where x0 drives failure (about 8% failures) and x1, x2 are noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(3.0 * x[:, 0] - 5.0)))
    frame = pd.DataFrame(x, columns=INPUTS)
    frame["label"] = (rng.uniform(size=n) < p).astype(int)
    frame["month"] = pd.PeriodIndex(rng.choice(pd.period_range(start, end, freq="M"), size=n), freq="M")
    return frame


@pytest.mark.skip(reason="waits for #18")
def test_winsorising_limits_come_from_the_training_block_only():
    from src.default_models import winsorise

    train = pd.DataFrame({"x0": np.arange(1.0, 101.0)})
    low, high = np.percentile(train["x0"], [5, 95])
    for extreme in (200.0, 2000.0):
        other = pd.DataFrame({"x0": [0.0, 50.0, extreme]})
        w_train, w_other = winsorise(train, other, ["x0"])
        assert w_other["x0"].tolist() == pytest.approx([low, 50.0, high])
        assert w_train["x0"].min() == pytest.approx(low) and w_train["x0"].max() == pytest.approx(high)


@pytest.mark.skip(reason="waits for #18")
def test_class_weight_is_negatives_over_positives():
    from src.default_models import class_weight

    assert class_weight(pd.Series([0] * 95 + [1] * 5)) == pytest.approx(19.0)


@pytest.mark.skip(reason="waits for #18")
def test_every_model_fits_gives_probabilities_and_applies_its_class_weight():
    from src.config import load_config
    from src.default_models import class_weight, ladder

    data = planted()
    weight = class_weight(data["label"])
    models = ladder(load_config(), pos_weight=weight)
    assert {"naive_bayes", "logit_chs", "logit_l1", "tree", "forest", "xgboost", "lightgbm", "mlp"} == set(models)
    for name, model in models.items():
        model.fit(data[INPUTS], data["label"])
        p = model.predict_proba(data[INPUTS])[:, 1]
        assert ((p >= 0) & (p <= 1)).all(), name
    assert models["xgboost"].get_params()["scale_pos_weight"] == pytest.approx(weight)
    assert models["lightgbm"].get_params()["scale_pos_weight"] == pytest.approx(weight)
    assert models["forest"].get_params()["class_weight"] == "balanced"


@pytest.mark.skip(reason="waits for #18")
def test_every_model_but_naive_bayes_finds_a_planted_signal():
    from src.config import load_config
    from src.default_models import class_weight, ladder, scores

    # The best possible ROC AUC here is about 0.94; a pruned tree is a step function and scores lower.
    train, test = planted(seed=1), planted(seed=2)
    for name, model in ladder(load_config(), pos_weight=class_weight(train["label"])).items():
        if name == "naive_bayes":
            continue
        model.fit(train[INPUTS], train["label"])
        bar = 0.75 if name == "tree" else 0.85
        assert scores(test["label"], model.predict_proba(test[INPUTS])[:, 1])["roc_auc"] > bar, name


@pytest.mark.skip(reason="waits for #19")
def test_time_blocks_for_test_year_2010():
    from src.default_models import time_blocks

    month = pd.Series(pd.period_range("1995-01", "2012-12", freq="M"))
    blocks = time_blocks(month, test_year=2010)
    years = {name: month[mask].dt.year for name, mask in blocks.items()}
    assert years["train"].max() == 2006 and month[blocks["train"]].max() == pd.Period("2006-12", "M")
    assert set(years["tune"]) == {2007} and set(years["calibrate"]) == {2008} and set(years["test"]) == {2010}
    assert sum(mask.astype(int) for mask in blocks.values()).max() == 1  # no month in two blocks
    assert not any(mask[month.dt.year == 2009].any() for mask in blocks.values())


@pytest.mark.skip(reason="waits for #19")
def test_every_fitting_outcome_is_known_before_the_test_year():
    from src.default_models import time_blocks

    month = pd.Series(pd.period_range("1980-01", "2025-12", freq="M"))
    for test_year in (2000, 2012, 2024):
        blocks = time_blocks(month, test_year=test_year)
        fitting = month[blocks["train"] | blocks["tune"] | blocks["calibrate"]]
        assert (fitting + 12).max() <= pd.Period(f"{test_year - 1}-12", "M")


@pytest.mark.skip(reason="waits for #19")
def test_scores_on_a_hand_computed_set():
    from src.default_models import scores

    out = scores(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert out["roc_auc"] == pytest.approx(0.75)
    assert out["pr_auc"] == pytest.approx(5 / 6)
    assert out["brier"] == pytest.approx(0.158125)


@pytest.mark.skip(reason="waits for #19")
def test_later_labels_do_not_change_the_recalibrated_probabilities():
    from sklearn.linear_model import LogisticRegression

    from src.default_models import fit_and_score

    data = planted(n=6000, start="2000-01", end="2010-12")
    later = data["month"].dt.year.isin([2009, 2010])  # the left-out year and the test year
    flipped = data.assign(label=np.where(later, 1 - data["label"], data["label"]))
    p1, _ = fit_and_score(LogisticRegression(), data, test_year=2010, inputs=INPUTS)
    p2, _ = fit_and_score(LogisticRegression(), flipped, test_year=2010, inputs=INPUTS)
    pd.testing.assert_series_equal(p1, p2)


@pytest.mark.skip(reason="waits for #19")
def test_tuning_keeps_the_setting_with_the_best_pr_auc_on_the_tune_block():
    from sklearn.tree import DecisionTreeClassifier

    from src.default_models import tune

    data = planted(n=6000, start="2000-01", end="2010-12")
    model = DecisionTreeClassifier(class_weight="balanced", random_state=0)
    best = tune(model, data, test_year=2010, inputs=INPUTS, grid={"max_depth": [1, 3]})
    assert best.get_params()["max_depth"] == 3  # PR AUC on 2007: about 0.29 at depth 1, 0.47 at depth 3
