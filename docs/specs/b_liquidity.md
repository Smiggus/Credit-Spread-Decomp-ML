# Stage b spec: liquidity measures in Spark

Milestone: [Stage b, liquidity measures in Spark](https://github.com/Smiggus/Credit-Spread-Decomp-ML/milestone/3). Issues #9 to #13.

## Why this stage

In stage a the five liquidity inputs (trade counts and volume on the snapshot day and in the month) took about 6% of the split and barely moved the model when left out. Either liquidity matters little for these spreads, or those inputs measure it badly. This stage builds the standard bond liquidity measures from every TRACE trade and reruns stage a with them, to tell the two apart.

## The result this stage must produce

1. **Monthly liquidity measures per bond**, from cleaned trade-level TRACE Enhanced, for January 2024 to March 2026, kept outside the repo with the other licensed data.
2. **`reports/b_liquidity_summary.csv`**: for each measure, by IG and HY and by month, the number of bonds with a value and the 25th, 50th and 75th percentiles. Totals only.
3. **Stage a rerun with the new measures** in the liquidity group: `reports/b_group_shares.csv` and `reports/b_drop_group.csv`, and a README table of liquidity's share and drop-group rise before and after.

The stage is done when the three exist, every stage b test passes with none skipped, and the README has a results section. Extending the measures back to 2002 (#13) comes after the results and does not block them.

## Measures

All are computed per bond and calendar month, from trades **before the snapshot day** of that month. The spread on the snapshot day comes from that day's trades, so using them for the inputs would explain the spread with itself.

| Measure | Definition | Source |
|---|---|---|
| Amihud | For each day with at least two trades, the mean of \|r_j\| / Q_j, where r_j is the price return from the previous trade that day and Q_j the trade's par volume in USD millions. Monthly value: the median over days. | Amihud (2002), in the bond form of Dick-Nielsen, Feldhütter and Lando (2012) |
| Imputed roundtrip cost (IRC) | Trades in a bond on one day with the same par volume form a roundtrip. IRC = (highest price - lowest price) / highest price. Daily value: the mean over that day's roundtrips. Monthly value: the median over days. | Feldhütter (2012), as used by Dick-Nielsen, Feldhütter and Lando (2012) |
| Amihud risk, IRC risk | The standard deviation of the daily values within the month | Dick-Nielsen, Feldhütter and Lando (2012) |
| Lambda | The sum of the four measures above, each standardised. The means and standard deviations come from the **train months only**, so the valid and test months do not shape the scale. | Dick-Nielsen, Feldhütter and Lando (2012) |
| Roll | 2 x sqrt(-cov(dP_j, dP_j-1)) over consecutive price changes in the month (sample covariance), in price points; 0 when the covariance is positive. At least three price changes. | Roll (1984); Bao, Pan and Wang (2011) |
| Zero-trade share | The share of the month's trading days (days with any TRACE trade) on which the bond did not trade | Standard |

A bond with too few trades for a measure gets an empty value. XGBoost handles empty values, and the summary reports how many bonds have each measure.

The earlier plan also listed a bid-ask spread from matched customer buys and sells. IRC measures the same cost from the same trades, so it is left out.

## Cleaning (#10)

The cleaning works on plain status names (trade, cancel, correction, reversal), mapped from the WRDS codes in one small function, so the logic and its tests do not depend on the codes. Follow Dick-Nielsen's filter for Enhanced TRACE: remove cancelled trades and the trades they cancel, replace corrected trades with their corrections, remove reversals and the trades they reverse, and keep one side of each interdealer trade so it is not counted twice. The exact WRDS field codes are checked against the WRDS TRACE manual in #10 and listed in this spec then. The cleaning reports how many trades each step removes, as totals.

## Where things run

- **Export (#9).** `scripts/b_export_trace.py` reads `trace.trace_enhanced` month by month with the read-only WRDS account and writes Parquet, one folder per month, to `TRACE_DIR`. Only the columns the cleaning and measures need. Row counts per month are logged, no rows printed.
- **Storage.** `TRACE_DIR` is a private bucket on the local MinIO object store. The export writes Parquet there, and the Spark cluster on the burst box reads the trades from it and writes the measures back as Parquet. A local path also works and is checked to be outside the repo.
- **Cleaning and measures (#10, #11)** run on the Spark Connect cluster, from `src/trace.py` and `src/liquidity.py`. They write one small bond-month table of measures to `LIQUIDITY_PATH`.
- **Join and rerun (#12)** run in pandas, as stage a does: the measures are joined to the panel on bond and month, and the liquidity group in `config.yaml` gains the new columns.

## Functions

| Function | Does | Issue |
|---|---|---|
| `scripts/b_export_trace.py` | Month-by-month export to Parquet; refuses a local path inside the repo | #9 |
| `trace.clean_trades(trades)` | Spark DataFrame in, cleaned trades out | #10 |
| `trace.cleaning_counts(trades)` | Trades removed by each cleaning step, as totals | #10 |
| `liquidity.daily_measures(trades, snapshot_days)` | Daily Amihud and IRC per bond, trades before the snapshot day only | #11 |
| `liquidity.monthly_measures(trades, snapshot_days)` | Every measure above per bond and month, except lambda | #11 |
| `liquidity.add_lambda(measures, train_months)` | Standardises on the train months and adds lambda | #11 |
| `liquidity.join_measures(panel, measures)` | Adds the measures to the panel on bond and month | #12 |

## Acceptance tests (`tests/test_trace.py`, `tests/test_liquidity.py`)

Written before the code, on small made-up trade sets with hand-computed answers (`tests/trades.py`), and skipped until their issue is done. They run on Spark: on a local Spark in CI, and on the cluster locally when `SPARK_REMOTE` is set. When the first Spark test is unskipped (#10), CI installs Java 17 and the full `pyspark==4.2.0` for a local Spark; this machine's Java 26 is not supported by Spark 4.2, which is why local runs use the cluster.

| Test | Issue |
|---|---|
| The export refuses a local path inside the repo | #9 |
| A cancelled trade and the trade it cancels are both removed | #10 |
| A corrected trade replaces the original | #10 |
| An interdealer trade reported by both dealers is counted once | #10 |
| Amihud on a hand-computed day | #11 |
| IRC: two same-size trades at 100 and 99 on one day give 0.01 | #11 |
| Roll: prices bouncing 100, 101, 100, 101, 100 give 2 x sqrt(4/3); a trend with positive covariance gives 0 | #11 |
| Zero-trade share: a bond trading on two of four market days gives 0.5 | #11 |
| Trades on the snapshot day are left out of every measure | #11 |
| Lambda's scale comes from the train months only | #11 |
| Joining the measures keeps every panel row and adds the measure columns | #12 |

## References

- Amihud (2002), "Illiquidity and stock returns: cross-section and time-series effects", *Journal of Financial Markets* 5, 31-56.
- Bao, Pan and Wang (2011), "The illiquidity of corporate bonds", *Journal of Finance* 66(3), 911-946.
- Dick-Nielsen, Feldhütter and Lando (2012), "Corporate bond liquidity before and after the onset of the subprime crisis", *Journal of Financial Economics* 103(3), 471-492.
- Dick-Nielsen, notes on cleaning Enhanced TRACE data (2009, 2014), the basis of the WRDS cleaned TRACE data.
- Feldhütter (2012), "The same bond at different prices: identifying search frictions and selling pressures", *Review of Financial Studies* 25(4), 1155-1206.
- Roll (1984), "A simple implicit measure of the effective bid-ask spread in an efficient market", *Journal of Finance* 39(4), 1127-1139.
