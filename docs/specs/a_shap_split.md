# Stage a spec: SHAP split by input group

Milestone: [Stage a, SHAP split by input group](https://github.com/Smiggus/Credit-Spread-Decomp-ML/milestone/2). Issues #4 to #8. #3, loading the panel, is done.

## The result this stage must produce

Built on the train and valid months only. The test months stay closed until the pricer.

1. **`reports/a_group_shares.csv`**. For each input group (bond terms, liquidity, company, stock and balance sheet, market): its mean contribution to the predicted spread in bp, and its share of the mean absolute contribution in %. Cut by IG and HY, by maturity bucket and by month, on the valid months. Totals only, with no bond or company identifiers.
2. **`reports/figures/a_group_shares.png`**. The shares by month, one panel for IG and one for HY.
3. **`reports/a_drop_group.csv`**. Valid MAE in bp for IG and HY: the full model, then the model refit without each group in turn.

The stage is done when:
- the three files exist;
- the model's valid MAE is close to the ML II raw-spread run (about 15 bp IG and 54 bp HY);
- no stage a test is skipped;
- the README has a short results section (#8).

## Definitions

- **Contribution.** XGBoost's own SHAP values (`pred_contribs=True` on the booster), summed over the inputs of each group, in bp. For each row, the base value plus the five group contributions equals the prediction, up to XGBoost's float32 rounding: about 1e-6 of the spread, 0.0012 bp at most on the real valid months.
- **Share.** Within a cut, a group's mean absolute contribution divided by the sum of all groups' mean absolute contributions. The shares in a cut sum to 100%.
- **Limit.** A contribution shows what the model uses, not what causes the spread. Correlated inputs, such as grade and distance to default, can trade credit between groups. The drop-group check is a second view of the same question.

## Functions (`src/shap_split.py`)

| Function | Does | Issue |
|---|---|---|
| `model_inputs(df)` | The 33 inputs, with categorical columns as pandas categories and booleans as 0/1 | #4 |
| `fit_spread_model(train)` | XGBoost regressor on the raw spread, settings from `config.yaml`. Raises ValueError on any test row | #4 |
| `predict_spread(model, df)` | Predicted spread in bp | #4 |
| `group_contributions(model, df)` | One column per input group plus `base`, in bp, row-aligned with `df` | #5 |
| `group_shares(contribs, df, by)` | For each cut in `by` and each group: `mean_bp` and `share_pct`. No identifier columns | #6 |
| `drop_group_mae(train, valid)` | Valid MAE in bp by stratum, for the full model (`left_out = "none"`) and for each group left out | #7 |

The chart and the CSV files are written by `scripts/a_shap_split.py`, which runs the stage end to end (#6, #7).

XGBoost settings, the same as the ML II final model: 500 trees, depth 10, learning rate 0.03, 80% of rows and 70% of columns per tree. They go in `config.yaml` with #4.

## Acceptance tests (`tests/test_shap_split.py`)

Written before the code, on synthetic panels where the answer is known. Each is skipped until its issue is done.

| Test | Issue |
|---|---|
| `fit_spread_model` refuses rows from the test months | #4 |
| On a synthetic panel, valid MAE beats predicting the train mean | #4 |
| Base plus group contributions equals the prediction, within float32 rounding (relative 1e-6, absolute 1e-3 bp) | #5 |
| A categorical input's effect lands in its own group (sector in company) | #5 |
| When a synthetic spread depends only on the liquidity inputs, liquidity's share is above 80% | #5 |
| Shares within each cut sum to 100% | #6 |
| The shares table has no bond or company identifier columns | #6 |
| Leaving out the group that drives a synthetic spread raises valid MAE the most | #7 |
