# Stage c spec: Merton and the default classifier ladder

Milestone: [Stage c, Merton and the classifier ladder](https://github.com/Smiggus/Credit-Spread-Decomp-ML/milestone/4). Issues #14 to #20. The survival, sequence and ensemble models (#21 to #23) have their own milestone and spec.

## Why this stage

Stages a and b describe what a spread model uses. They do not say how much of the spread pays for default. This stage builds the default part three ways, from the Merton model, from Merton calibrated to historical default rates, and from a machine learning default model trained on decades of listed companies. It then splits each bond's spread into a default part, a liquidity part, a market part and the rest. The calibrated Merton spread should reproduce the credit spread puzzle: a small default share in IG and a larger one in HY. Whether the ML default model explains more of the spread is the open question.

Stage b found that leaving out the stock and balance sheet group costs HY less once the liquidity measures are in (5.6 bp against 10.2 bp). The split in #20 shows whether the measures carry default information.

## The result this stage must produce

Built on the panel's train and valid months. The test months stay closed until the pricer.

1. **Merton spreads** per bond-month (#14), and the calibrated Merton spread by IG and HY and maturity (#15). `reports/c_merton.csv`: the mean and quartiles of each model spread and of the observed spread, by IG and HY and maturity bucket, with each model's default share (mean model spread over mean observed spread).
2. **Default labels and the CRSP-Compustat match** (#16, #17). `reports/c_failures.csv` (failures per year by source) and `reports/c_match_rate.csv` (Compustat companies matched to CRSP per year, by method). Hand-checked samples go in the PR text, with names only for companies whose failure is public record.
3. **The classifier ladder scored on rolling test years** (#18, #19). `reports/c_model_scores.csv`: PR AUC, ROC AUC and Brier score per model and test year, and `reports/figures/c_calibration.png`.
4. **The split** (#20). `reports/c_split.csv`: for IG and HY, in bp and as a share of the mean spread, the default part from each default model, the liquidity part, the market part, the common part and the unexplained rest. A README table compares the default share from Merton, calibrated Merton and the best ML model.

All tables are totals or quartiles over at least 5 companies. No company or bond rows.

The stage is done when the four results exist, every stage c test passes with none skipped, and the README has a results section.

## The Merton spread (#14)

For each company and month in the panel, solve the two Merton equations for the asset value V and asset volatility sigma_V:

- E = V N(d1) - F exp(-rT) N(d2)
- sigma_E E = N(d1) sigma_V V

with d1 = (ln(V / F) + (r + sigma_V^2 / 2) T) / (sigma_V sqrt(T)) and d2 = d1 - sigma_V sqrt(T).

| Input | From the panel |
|---|---|
| Equity value E | `exp(log_market_cap)`, USD |
| Equity volatility sigma_E | `vol_252d`, annualised |
| Debt F | `total_debt`, USD, the latest balance sheet published before the snapshot. The panel has no split between short and long-term debt, so the usual default point (short-term debt plus half the long-term) is not available. |
| Rate r for solving | `ust_2y_pct`. The panel has no one-year yield. |
| Horizon T for solving | 1 year |

The panel's Treasury yields are in percent with semiannual compounding, so each is turned into a continuous rate as r = 2 ln(1 + y / 200), where y is the yield in percent.

The bond's model spread then uses the solved V and sigma_V with the bond's own maturity T_b (`years_to_maturity`) and rate r_b (`ust_yield_at_maturity_pct`):

s = -(1 / T_b) ln( N(d2) + V / (F exp(-r_b T_b)) N(-d1) ), in bp.

A company-month with no debt gets a spread of 0. One with a missing input, or where the solver does not converge, gets an empty spread, and the count of each goes in the report.

## Merton calibrated to historical default rates (#15)

Huang and Huang (2012) pick each rating's asset volatility so that the model's default probability matches the historical default rate, then compute the spread. This stage uses the shorter route of Chen, Collin-Dufresne and Goldstein (2009): in the Merton model, the default probability under the pricing measure is the historical one shifted by the asset Sharpe ratio lambda, so the historical default rate and lambda give the spread without solving for volatility. The panel has only IG and HY (`grade_bucket`), not letter ratings, so the calibration is by IG and HY and maturity, against the published cumulative default rates of investment-grade and speculative-grade issuers.

- PD_Q(T) = N( N^-1(PD_P(T)) + lambda sqrt(T) )
- s(T) = -(1 / T) ln( 1 - PD_Q(T) (1 - R) )

PD_P(T) is the historical cumulative default rate at horizon T, interpolated linearly between the published horizons and from zero at horizon 0. `default_rate` refuses a maturity past the last published horizon; on the panel, such a bond gets an empty calibrated spread and is counted in the report. R is the recovery rate. The default rates, the recovery rate and lambda go in `config.yaml` in #15, each with its source table named next to it. Lambda follows Chen, Collin-Dufresne and Goldstein's calibration from the equity premium.

## Default labels (#16)

The unit is a company-month of a common stock (CRSP's ordinary common shares; funds, units and ADRs are left out). The label is 1 when the company fails in any of the next 12 months (t + 1 to t + 12), else 0. Rows from the failure month on are dropped. A company that leaves for another reason, such as a merger, keeps label 0 and its rows end when it leaves. When the 12-month window runs past the last month of the delivery, the label is left empty, because the outcome is not yet known.

A failure is either of:

- a CRSP delisting for poor performance: in the old CRSP codes, the ones Shumway (2001) uses, 500 and 520 to 584. The 400s (liquidations) are left out because many are voluntary. The CIZ files code delistings differently, so #16 maps these to the CIZ fields from CRSP's conversion guide and lists the mapping in `config.yaml`;
- a Compustat deletion for bankruptcy (`dlrsn` 02) or liquidation (`dlrsn` 03) in `comp.company`.

When both sources record a failure for one company, it is dated by the earlier of the two. Compustat's deletion date can come months after the filing. `reports/c_failures.csv` shows how many failures each source and code adds. Only companies with CRSP stock data can be scored, because the model inputs need it.

## Matching CRSP stocks to Compustat companies (#17)

The CRSP-Compustat link table is blocked on our account. Compustat holds only each company's current CUSIP and ticker, while CRSP keeps the history of both for each stock. So the match finds the CRSP stock that ever carried Compustat's current identifier, and uses dates only to stay within that stock's life:

1. **CUSIP.** The first 8 digits of Compustat's CUSIP against every CUSIP in a CRSP stock's history. A company-month matches a stock that ever carried the CUSIP and was trading in that month. A CUSIP that passed to another company later matches each company in its own months.
2. **Ticker.** Where the CUSIP finds nothing, Compustat's ticker against the CRSP ticker in use in the company's last Compustat month, when its current ticker applies. That stock is then used for the company's months within the stock's life. Ticker matches are reported apart from CUSIP matches.
3. **One stock per company.** When a company has more than one CRSP stock in a month (share classes), returns and price come from the one with the larger market value, and market equity is the sum over all of them.

The match rate per year and method goes to `reports/c_match_rate.csv`, and a sample of 30 matches is checked by hand in the PR.

## Model inputs

The Campbell, Hilscher and Szilagyi (2008) inputs, per company-month:

| Input | Definition |
|---|---|
| NIMTAAVG | Net income over the market value of total assets (book liabilities plus market equity), averaged over the last four quarters with weights 0.533, 0.267, 0.133 and 0.067 (CHS's phi = 2^(-1/3) per month) |
| TLMTA | Total liabilities over the market value of total assets |
| EXRETAVG | Log excess return over the S&P 500, averaged over the last 12 months with weights proportional to phi^(j-1) |
| SIGMA | Annualised volatility of daily returns over the last three months, centred at zero as CHS do: sqrt(252 / (N - 1) x the sum of squared daily returns) |
| RSIZE | Log of market equity over the total market value of the S&P 500 |
| CASHMTA | Cash and short-term investments over the market value of total assets |
| MB | Market to book equity, with book equity raised by 10% of the gap to market equity, as CHS do; a negative book equity is set to $1 first |
| PRICE | Log share price, capped at log(15) |

Market equity is summed over all of a company's common stocks. Quarterly accounting data is used from its report date (`rdq`), or 90 days after the quarter ends when there is no report date. Each input is winsorised at the 5th and 95th percentiles of the training block only. Rows missing any input are dropped from the ladder; the count and the failure rate before and after the drop are reported, since dropping rows can remove failing companies more often than others. S&P ratings (`comp.adsprate`) stop in 2017 and the panel months have none, so they are left out of the ladder; one extra fit on the test years up to 2017 reports what they add.

SIGMA needs daily CRSP returns. That is the one step in this stage that runs on Spark: the three-month volatility is computed on the cluster and only the monthly value comes back.

## The ladder (#18)

All models see the same rows and inputs. All but Naive Bayes use class weights, and none resamples across time. Every model with randomness takes its seed from `config.yaml`.

| Model | Library | Class weights |
|---|---|---|
| Naive Bayes (Gaussian) | scikit-learn | None; the prior is the training failure rate |
| Logit on the CHS inputs | scikit-learn | Balanced |
| Logit with an L1 penalty on all inputs | scikit-learn | Balanced |
| Decision tree, pruned by cost-complexity | scikit-learn | Balanced |
| Random forest | scikit-learn | Balanced |
| XGBoost | xgboost | `scale_pos_weight` = negatives over positives |
| LightGBM | lightgbm | The same |
| MLP with dropout and early stopping | PyTorch | Positive weight in the loss = negatives over positives |

"All inputs" for the L1 logit means the CHS inputs, their squares, and the extra inputs added in #18 (for example the one-year return and market beta), listed in `config.yaml`. Hyperparameters come from a small grid per model in `config.yaml`: each setting is fitted on the train block and the one with the best PR AUC on the tune block is kept. The MLP, XGBoost and LightGBM also use the tune block for early stopping.

The libraries each issue needs (scipy, scikit-learn, lightgbm, PyTorch) are added to `environment.yml` and to CI's install line in that issue.

## Rolling time split and scoring (#19)

At the start of test year y, only outcomes known by the end of year y - 1 may be used. So, for test year y:

| Block | Months | Used for |
|---|---|---|
| Train | Up to December of y - 4 | Fitting |
| Tune | Year y - 3 | Hyperparameters and early stopping |
| Calibrate | Year y - 2 | Recalibrating the probabilities (isotonic regression) |
| (left out) | Year y - 1 | Its 12-month outcomes end in year y |
| Test | Year y | Scoring |

The window rolls forward one year at a time. Test years run from 2000 to 2024; 2024's outcomes end in December 2025, where CRSP ends. Test year 2025 is fitted the same way so the panel's 2025 months have a model, but it is not scored, because its outcomes are not known yet.

Each test year is scored with PR AUC (average precision), ROC AUC and the Brier score. The two AUCs use the model's raw scores, since isotonic recalibration creates ties that change the ranking; the Brier score and the calibration plot use the recalibrated probabilities. The calibration plot pools all test years by decile of predicted probability. Accuracy is not reported. Failures are about 1% of company-years, so a model that always says "no failure" is 99% accurate.

## The split (#20)

1. **Join the panel to CRSP.** Each panel bond joins to its company's CRSP stock through `wrdsapps.bondcrsp_link` on CUSIP and date. The result is checked against the data platform's issuer-to-ticker map for the panel's companies, and disagreements are counted and left out.
2. **Default spread.** For the ML models, the 12-month default probability from the model fitted for that month's year. A probability is a real-world quantity, while the Merton spreads are prices that include a premium for default risk. So the ML default spread is put on the same footing with the same shift as #15: calibrated_spread(PD, lambda, 1, R). The real-world expected loss, PD x (1 - R), is reported next to it, and the gap between the two is the default risk premium. For Merton and calibrated Merton, their spreads from #14 and #15.
3. **Residual model.** For each default model, the residual (observed spread minus default spread) is fitted with stage a's XGBoost settings on the train months, on the liquidity inputs (with the stage b measures) and the market inputs only. The groups are listed under `split.residual_groups` in `config.yaml`, added in #20. Its SHAP values split each prediction into a common part (the base value), a liquidity part and a market part.
4. **Accounting.** For every bond-month, spread = default + common + liquidity + market + unexplained, where unexplained is the residual minus the sum of the three SHAP parts. `reports/c_split.csv` gives the mean of each part in bp and as a share of the mean spread, by IG and HY, train and valid months apart. The deep residual model fits its train months almost exactly, so the unexplained part is read on the valid months.

The common part is the level the residual model cannot pin on either group. It is reported as its own line, not shared out.

## Where things run

- **Export (#16, #17), done by the data platform.** The platform exports the WRDS tables below with the read-only account, under the same contract as the TRACE delivery: Parquet in its private bucket, one `manifest.json` per table (or per year for the daily file) with the row count, the columns and the WRDS query. A pull runs only after the owner's go-ahead.
- **Storage.** `COMPANY_DATA_DIR` (in `.env`) points at the delivery. The repo reads it through Spark, checks it against the manifests, selects the columns it needs, and writes them as Parquet to `COMPANY_CACHE_DIR`, a folder outside the repo on this machine. Everything after that runs in pandas.
- **Merton (#14, #15)** runs in pandas on the panel and needs no new data.

### Data contract for the company data (#16, #17)

| Table | For |
|---|---|
| `crsp.stkmthsecuritydata` | Monthly returns, prices and shares outstanding |
| `crsp.stkdlysecuritydata` | Daily returns for SIGMA (the large one; partitioned by year) |
| `crsp.stkdelists` | Delisting dates and reasons |
| `crsp.stksecurityinfohist` | CUSIP and ticker history with dates, for the match |
| CRSP's monthly S&P 500 index series | The S&P 500 return and total market value |
| `comp.fundq`, `comp.funda` | Quarterly and annual balance sheets and income |
| `comp.company` | Deletion reason and date |
| `comp.adsprate` | S&P issuer ratings to 2017, for the one extra fit |
| `wrdsapps.bondcrsp_link` | Bond CUSIP to CRSP stock, for #20 |

Window: January 1980 to the latest month on WRDS. Every column as WRDS serves it, no filtering. The access check before the pull confirms that `stksecurityinfohist` and the index series are readable; the plan's November check did not list them. The columns each step needs go in `config.yaml` when the reader is written, and a delivery without them is refused, as in stage b.

## Functions

| Function | Does | Issue |
|---|---|---|
| `merton.solve_assets(equity, equity_vol, debt, rate, horizon=1.0)` | V and sigma_V for one company-month; empty when an input is missing or the solver fails; V = E and sigma_V = 0 with no debt | #14 |
| `merton.model_spread(assets, asset_vol, debt, rate, maturity)` | The bond's Merton spread in bp | #14 |
| `merton.panel_spreads(panel)` | Solves each company-month once and adds `merton_spread_bps` to every bond row; returns the counts of empty rows by reason | #14 |
| `merton.risk_neutral_pd(pd, sharpe, maturity)` | PD_Q from PD_P | #15 |
| `merton.calibrated_spread(pd, sharpe, maturity, recovery)` | The calibrated spread in bp | #15 |
| `merton.default_rate(table, bucket, maturity)` | The cumulative default rate at a maturity, interpolated; refuses one past the table | #15 |
| `delivery.check_delivery(...)` | The stage b delivery check, moved from `trace` to `src/delivery.py` so both stages use it | #16 |
| `company.failures(delistings, deletions, codes)` | One failure date per company, the earlier source winning | #16 |
| `company.labels(months, failures, horizon=12, last_month=None)` | The 12-month label per company-month; drops rows from the failure on; empty where the window runs past `last_month` | #16 |
| `company.match_crsp(compustat, names, market_values)` | Compustat company to the CRSP stock that ever carried its CUSIP (then its ticker), within the stock's life, with the method; one stock per company-month, the larger by market value | #17 |
| `company.chs_inputs(monthly, quarterly, index, sigma)` | The eight CHS inputs per company-month, accounting data from its report date | #18 |
| `company.nimta_avg(values)`, `company.available_from(quarters)`, `company.price_input(price)`, `company.adjusted_book(book, market)` | The pieces of the CHS inputs that the tests check by hand | #18 |
| `default_models.time_blocks(months, test_year)` | Train, tune, calibrate and test months for one test year | #19 |
| `default_models.winsorise(train, other, cols)` | Limits from the training block, applied to both | #18 |
| `default_models.class_weight(y)` | Negatives over positives | #18 |
| `default_models.ladder(config, pos_weight)` | The eight models with their class weights, by name | #18 |
| `default_models.tune(model, frame, test_year, inputs, grid)` | Fits each grid setting on the train block and keeps the best by PR AUC on the tune block | #19 |
| `default_models.fit_and_score(model, frame, test_year, inputs)` | Fits on the train block, recalibrates on the calibrate block, and returns the test block's recalibrated probabilities and its scores (AUCs on the raw scores) | #19 |
| `default_models.scores(y, p)` | PR AUC (average precision), ROC AUC and Brier score | #19 |
| `split.default_spread(pd, recovery)` | The real-world expected loss PD x (1 - R) in bp; the priced version is `merton.calibrated_spread` | #20 |
| `split.split_parts(panel, default_col, config)` | The residual model and the parts per bond-month, on the panel's index; the groups come from `split.residual_groups` in `config.yaml`, and any group other than liquidity and market is refused | #20 |
| `split.split_table(parts, panel)` | Rows and mean parts and shares by IG and HY and partition, no identifiers | #20 |

## Acceptance tests

Written before the code and skipped until their issue is done. Imports sit inside the tests so the modules load before `src/merton.py`, `src/delivery.py`, `src/company.py`, `src/default_models.py` and `src/split.py` exist.

| Test | Issue |
|---|---|
| Hull's example (E = 3, sigma_E = 80%, r = 5%, F = 10, T = 1) gives V = 12.40, sigma_V = 21.23% and a default probability of 12.7% | #14 |
| The spread from the solved assets matches the debt's value, V - E: 123.7 bp in Hull's example | #14 |
| The spread rises with debt and with asset volatility | #14 |
| No debt gives a spread of 0; a missing input gives an empty spread, not an error | #14 |
| Each company-month is solved once, and every bond of it gets its own maturity's spread | #14 |
| With a zero Sharpe ratio the pricing-measure default probability equals the historical one: PD 1%, R = 40%, T = 1 gives 60.2 bp | #15 |
| PD_P = N(-2), lambda = 0.5, T = 4, R = 40% gives PD_Q = N(-1) = 15.87% and 250.1 bp | #15 |
| The default rate halfway between two published horizons is the average of the two, a maturity before the first is interpolated from zero, and one past the table is refused | #15 |
| A CRSP failure four months before Compustat's deletion dates the failure at the CRSP date | #16 |
| A Compustat deletion for a merger is not a failure | #16 |
| Month t is labelled 1 when the failure falls in t + 1 to t + 12, 0 otherwise; rows from the failure month on are dropped | #16 |
| A company that leaves for a merger keeps label 0 and its rows end | #16 |
| The label is empty when the 12-month window runs past the last month | #16 |
| A CUSIP that passed to another company matches each company in its own months | #17 |
| A company whose CUSIP changed in CRSP's history still matches through its current CUSIP | #17 |
| The ticker is used only when the CUSIP finds nothing, and its matches are counted apart | #17 |
| With two share classes, the larger by market value is kept | #17 |
| NIMTAAVG of quarterly values 0.04, 0.02, 0 and -0.02 (latest first) is 0.0253 | #18 |
| Accounting data is used from its report date, or 90 days after the quarter without one | #18 |
| PRICE is capped at log(15); MB uses the adjusted book equity | #18 |
| Winsorising limits come from the training block only | #18 |
| The class weight is negatives over positives | #18 |
| Every model in the ladder fits, predicts probabilities between 0 and 1, and applies its class weights | #18 |
| On made-up data with a planted signal, every model except Naive Bayes reaches a ROC AUC above 0.9 (the pruned tree, a step function, above 0.85) | #18 |
| For test year 2010: train to December 2006, tune 2007, calibrate 2008, test 2010; no month in two blocks | #19 |
| Every fitting row's 12-month outcome is known by the end of the year before the test year | #19 |
| Scores on a small hand-computed set match | #19 |
| Changing the labels of the test year and the year before does not change the recalibrated probabilities | #19 |
| Tuning keeps the grid setting with the best PR AUC on the tune block | #19 |
| PD 2% and R = 40% give a default spread of 120 bp | #20 |
| For every train and valid bond-month the parts add up to the spread | #20 |
| The residual model refuses an input outside the liquidity and market groups | #20 |
| The split table has no identifiers and covers every row | #20 |

## References

- Bharath and Shumway (2008), "Forecasting default with the Merton distance to default model", *Review of Financial Studies* 21(3), 1339-1369.
- Campbell, Hilscher and Szilagyi (2008), "In search of distress risk", *Journal of Finance* 63(6), 2899-2939.
- Chen, Collin-Dufresne and Goldstein (2009), "On the relation between the credit spread puzzle and the equity premium puzzle", *Review of Financial Studies* 22(9), 3367-3409.
- Hull, *Options, Futures, and Other Derivatives*, the chapter on credit risk: the worked example of estimating default probability from equity prices.
- Huang and Huang (2012), "How much of the corporate-Treasury yield spread is due to credit risk?", *Review of Asset Pricing Studies* 2(2), 153-202.
- Merton (1974), "On the pricing of corporate debt: the risk structure of interest rates", *Journal of Finance* 29(2), 449-470.
- Shumway (2001), "Forecasting bankruptcy more accurately: a simple hazard model", *Journal of Business* 74(1), 101-124.
