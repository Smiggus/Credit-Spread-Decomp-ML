# Credit Spread Decomposition and ML Bond Pricer

Splits US corporate bond spreads into default, liquidity and risk premium parts using TRACE data, liquidity measures built in Spark, an ML default model, a Merton structural model, a company network in GraphFrames and CDS, then prices bonds from the Treasury curve plus an ML spread forecast.

## The question

How much of a US corporate bond's spread pays for expected default, how much pays for poor liquidity, and how much is a market-wide risk premium? How do those shares differ between investment grade and high yield, and how do they move over time?

The spread is the bond's yield minus the yield of the US Treasury with the same maturity, in basis points (100 bp = 1%). Structural models of default explain only part of it, above all for safe companies. This gap is known as the credit spread puzzle.

## Stages

Each stage is a [milestone](https://github.com/Smiggus/Credit-Spread-Decomp-ML/milestones), and each task in it is an issue. Stages run in this order:

| Step | Stage | What | Status |
|---|---|---|---|
| 1 | (a) | SHAP split of a gradient boosting spread model by input group | Done |
| 2 | (b) | Liquidity measures from trade-level TRACE in Spark | Not started |
| 3 | (c) | Merton model and a ladder of default classifiers, from Naive Bayes and logit to trees, boosting and an MLP | Not started |
| 3b | (c) | Survival, sequence and ensemble default models | Not started |
| 4 | (h) | 10-K and 10-Q parsing from SEC EDGAR: subsidiaries, default warnings, debt maturities, text tone | Not started |
| 5 | (d) | Company network in Spark GraphFrames: customer, supplier and parent links, ratings and contagion | Not started |
| 6 | (f) | Bond pricer: Treasury curve plus the predicted spread | Not started |
| 7 | (e) | CDS as the default part, bond spread minus CDS as the rest | Not started |
| 8 | (i) | LLM extraction from filings, checked against hand labels, and an explanation layer on the pricer | Not started |
| 9 | (g) | Dealer network, if dealer-level TRACE data ever becomes available | Off |

Each stage gets a short results section here when it is done.

## Results

### Stage a: what the spread model uses

An XGBoost model of the spread (500 trees, depth 10), trained on January 2024 to May 2025 and scored on July to October 2025, misses by 15.3 bp in IG and 50.3 bp in HY on average. Its SHAP values split each predicted spread into the five input groups. The split explains the model, not the market: it says which inputs the model relies on, and correlated inputs can trade credit between groups. Spec: [`docs/specs/a_shap_split.md`](docs/specs/a_shap_split.md).

![Each input group's share of the mean absolute SHAP contribution, by month](reports/figures/a_group_shares.png)

Share of the mean absolute contribution on the valid months, and the rise in valid error when the group is left out and the model refit:

| Input group | IG share | IG error rise | HY share | HY error rise |
|---|---|---|---|---|
| Bond terms (grade, maturity, coupon, seniority) | 40.5% | +6.5 bp | 51.5% | +5.8 bp |
| Stock and balance sheet | 33.1% | +1.7 bp | 29.6% | +10.2 bp |
| Company (sector, size of the bond program) | 11.8% | -0.1 bp | 9.2% | +4.0 bp |
| Market (Treasury curve, index spreads, VIX) | 7.7% | +2.1 bp | 3.7% | +3.7 bp |
| Liquidity (trade counts and volume) | 6.9% | -0.2 bp | 6.1% | +0.4 bp |

What this shows:

- **Bond terms carry the largest share**, about two fifths of the split in IG and half in HY. Leaving them out also removes the IG or HY grade itself, which is why IG error rises most.
- **In HY, the company's stock and balance sheet matter most for accuracy.** Leaving that group out costs 10.2 bp, more than leaving out bond terms (5.8 bp). The model can recover much of the rating from other inputs, but not the stock's volatility, leverage and distance to default. That is default risk, priced from equity.
- **The market share moves with the market.** It is 2.5% of the IG split in March 2024 and 24.1% in April 2025, the month of the tariff sell-off. HY shows the same peak at 10.5%.
- **The three simple liquidity inputs add almost nothing.** About 6% of the split, and leaving them out does not raise error (the small IG gains from leaving out liquidity or company are noise). Stage b replaces them with proper liquidity measures from trade-level TRACE, to test whether liquidity is really this small or only measured badly.

Tables: [`reports/a_group_shares.csv`](reports/a_group_shares.csv) (by grade, maturity and month, train months marked in-sample) and [`reports/a_drop_group.csv`](reports/a_drop_group.csv). Reproduce with `python -m scripts.a_shap_split` (about 8.5 minutes).

## Data

The market data is licensed and is not in this repo. It comes from WRDS (TRACE Enhanced, CRSP, Compustat and Compustat customer segments) and a one-off Bloomberg CDS pull. Company filings come from SEC EDGAR, which is public. The code reads data from a local folder set in `.env`, and refuses any data path inside the repo. Results in this repo are totals and charts only.

## Layout

```
src/            package code (run modules from the repo root: python -m src.<module>)
  config.py     settings from config.yaml, paths from .env
  spark.py      builds the Spark session (Spark Connect or local)
scripts/        one-off jobs such as data exports
notebooks/      one notebook per stage, exploration only
reports/        write-up and figures (totals only)
docs/           method notes per stage
tests/          tests
config.yaml     project settings, safe to commit
.env.example    the machine-specific settings .env needs
```

## Setup

```
conda env create -f environment.yml
conda activate credit-spread-decomp
cp .env.example .env     # then fill in the paths
pytest
```

The Spark stages target Spark 4.2.0 on Scala 2.13 with GraphFrames 0.12.3. With a Spark Connect server, set `SPARK_REMOTE`; GraphFrames must be installed on the server. To run Spark locally instead, install `pyspark==4.2.0`, use Java 17 or 21, and set `GRAPHFRAMES_PACKAGE` to the GraphFrames build for Spark 4.x and Scala 2.13.

## How work is done

- `main` holds releases and is protected: changes reach it only by pull request from `dev`, with CI passing.
- `dev` is the working branch. Each issue gets a branch from `dev` (`feature/`, `docs/`, `chore/` or `ci/` plus a short name) and a pull request back into `dev`.
- PRs into `dev` do not close issues automatically, so close each issue by hand after its merge.
- **Spec first.** Each stage starts with a spec in `docs/specs/`: the result the stage must produce, the functions it needs and their acceptance tests. The acceptance tests are committed with the spec and skipped until their issue is done, so the skip ledger shows what is left. A stage is done when its result is produced and none of its tests are skipped.
- **Review every PR.** Before merging, the diff is read as a reviewer for unnecessary complexity, bugs, readability and tests, and the findings go in the PR's Review section. The topic is complex; the code should not be.
- **CI** (`lint-and-test`) runs on every pull request and push to `dev` and `main`: flake8, isort, radon (no function above complexity 10), pylint (10.00) and pytest with at least 90% branch coverage of `src/`.

## Licence

All rights reserved. See `LICENSE`.
