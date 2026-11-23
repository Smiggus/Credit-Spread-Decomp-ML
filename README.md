# Credit Spread Decomposition and ML Bond Pricer

Splits US corporate bond spreads into default, liquidity and risk premium parts using TRACE data, liquidity measures built in Spark, an ML default model, a Merton structural model, a company network in GraphFrames and CDS, then prices bonds from the Treasury curve plus an ML spread forecast.

## The question

How much of a US corporate bond's spread pays for expected default, how much pays for poor liquidity, and how much is a market-wide risk premium? How do those shares differ between investment grade and high yield, and how do they move over time?

The spread is the bond's yield minus the yield of the US Treasury with the same maturity, in basis points (100 bp = 1%). Structural models of default explain only part of it, above all for safe companies. This gap is known as the credit spread puzzle.

## Stages

Each stage is a [milestone](https://github.com/Smiggus/Credit-Spread-Decomp-ML/milestones), and each task in it is an issue. Stages run in this order:

| Step | Stage | What | Status |
|---|---|---|---|
| 1 | (a) | SHAP split of a gradient boosting spread model by input group | Not started |
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
- CI (`lint-and-test`) runs flake8 and pytest on every pull request and push to `dev` and `main`.

## Licence

All rights reserved. See `LICENSE`.
