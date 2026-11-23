# Credit Spread Decomposition and ML Bond Pricer

Splits US corporate bond spreads into default, liquidity and risk premium parts using TRACE data, liquidity measures built in Spark, an ML default model, a Merton structural model, a company network in GraphFrames and CDS, then prices bonds from the Treasury curve plus an ML spread forecast.

## The question

How much of a US corporate bond's spread pays for expected default, how much pays for poor liquidity, and how much is a market-wide risk premium? How do those shares differ between investment grade and high yield, and how do they move over time?

The spread is the bond's yield minus the yield of the US Treasury with the same maturity, in basis points (100 bp = 1%). Structural models of default explain only part of it, above all for safe companies. This gap is known as the credit spread puzzle.

## Stages

| Stage | What | Status |
|---|---|---|
| (a) | SHAP split of a gradient boosting spread model by input group | Not started |
| (b) | Liquidity measures from trade-level TRACE in Spark | Not started |
| (c) | Merton model and a ladder of default classifiers (Naive Bayes, logit, trees, boosting, neural networks, survival and sequence models), then ML on the residual spread | Not started |
| (d) | Company network in Spark GraphFrames: customer and supplier links, ratings and contagion | Not started |
| (e) | CDS as the default part, bond spread minus CDS as the rest | Not started |
| (f) | Bond pricer: Treasury curve plus the predicted spread | Not started |
| (h) | 10-K and 10-Q parsing from SEC EDGAR: subsidiaries, default warnings, debt maturities, text tone | Not started |
| (i) | LLM extraction from filings, checked against hand labels, and an explanation layer on the pricer | Not started |

Each stage gets a short results section here when it is done.

## Data

The market data is licensed and is not in this repo. It comes from WRDS (TRACE Enhanced, CRSP, Compustat and Compustat customer segments) and a one-off Bloomberg CDS pull. Company filings come from SEC EDGAR, which is public. The code reads it from a local folder set in `.env`. Results in this repo are totals and charts only.

## Layout

```
src/csd/        package code
  config.py     reads paths and settings from .env
  spark.py      builds the Spark session
notebooks/      one notebook per stage
reports/        write-up and figures (totals only)
tests/          tests
```

## Setup

```
python -m venv .venv
.venv/Scripts/activate        # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev,connect]"   # or [dev,local] to run Spark on this machine
cp .env.example .env              # then fill in the paths
```

The Spark stages target Spark 4.2.0 on Scala 2.13 with GraphFrames 0.12.3. With a Spark Connect server, set `SPARK_REMOTE`; GraphFrames must be installed on the server, and the client needs only the `connect` extra. To run Spark locally instead, install the `local` extra, use Java 17 or 21, and set `GRAPHFRAMES_PACKAGE` to the GraphFrames build for Spark 4.x and Scala 2.13.
