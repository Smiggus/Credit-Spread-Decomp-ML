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

Dick-Nielsen's filter for Enhanced TRACE, as ported in Tidy Finance's `clean_enhanced_trace`, on the WRDS columns. TRACE moved to a new platform on 6 February 2012 and changed its status codes (`trc_st`; FINRA's post-2012 file layout: T trade report, X trade cancel, C cancelled correction, R new correction, Y reversal). The window here is entirely after that date, so only those rules are built now; reports from before it are refused until #13 adds the older rules.

| Step | Rule |
|---|---|
| Trade reports | Keep T and R reports. R is the corrected version of a trade. |
| Cancels | An X or C report carries the same `msg_seq_nb` and trade details as the report it removes. Remove the trade matching on `msg_seq_nb` and the details: bond, volume, price, side, counterparty, execution date and time. No report date is used, so a cancel reported days later still finds its trade. |
| Reversals | A Y report's `orig_msg_seq_nb` is the `msg_seq_nb` of the trade it removes; match on that and the same details. Only that trade goes, even if an identical trade exists. |
| Interdealer trades | Both dealers report the trade. Drop a dealer buy report when a dealer sell report matches it on bond, execution date, volume and price; keep unmatched dealer buys. |
| Special and late | Keep only `wis_fl = N` (not when-issued), no special trade flag, no as-of flag, and settlement at most 7 days after execution. |

Empty values match each other in the joins, as in the reference code. The cleaning reports how many reports each step removes, as totals.

## Where things run

- **Export (#9), done by the data platform.** The data platform owns the WRDS account, the export and the pipeline; this repo works with what it delivers. The platform exports `trace.trace_enhanced` and `trace.trace_btds144a_enhanced` with the read-only account to Parquet in its private raw-data bucket, under the data contract below. This repo's part of #9 is the contract and a reader that checks a delivery against it.
- **Storage and access.** `TRACE_DIR` (in `.env`) points at the delivery. Only the Spark Connect server holds the bucket's credentials, so this repo reads the delivery through Spark and keeps no keys. The cluster writes the measures back as Parquet to `LIQUIDITY_PATH`.
- **Cleaning and measures (#10, #11)** run on the Spark Connect cluster, from `src/trace.py` and `src/liquidity.py`.
- **Join and rerun (#12)** run in pandas, as stage a does: the measures are joined to the panel on bond and month, and the liquidity group in `config.yaml` gains the new columns.

### Data contract for the TRACE delivery (#9)

| Item | Contract |
|---|---|
| Tables | `trace.trace_enhanced` and `trace.trace_btds144a_enhanced`, every column as WRDS serves it, no filtering or cleaning (cleaning happens here, in #10, so it is tested and visible) |
| Window | January 2024 to March 2026 first; back to July 2002 later (#13) |
| Layout | `<TRACE_DIR>/<table>/year=YYYY/month=MM/*.parquet`, partitioned on the trade execution date; days without trades have no file |
| Manifest | One `manifest.json` per month, written last: at least `table`, `month` ("YYYY-MM"), `rows`, and `columns` (name, WRDS type, Arrow type), plus the export time and the WRDS query. The name has no leading underscore because Spark skips such files, and this repo reads the manifests through Spark |
| Types | WRDS `numeric` arrives as float64, as the `wrds` package returns it; missing values as null. Rows with no trade execution date are not exported |
| Licence | Private bucket, no public access; no rows in messages, logs or the repo |

The reader gathers each month's manifest and Spark's own row count of the month's files, on the cluster, and passes both to `check_delivery`, a plain function with no Spark or file access. It fails with a message naming the month or column when a month is missing, a row count differs from its manifest, or a column the cleaning needs is absent.

## Functions

| Function | Does | Issue |
|---|---|---|
| `trace.check_delivery(manifests, counts, months, required)` | Plain check, no Spark or files: every month present, row counts match the manifests, required columns listed in the manifests | #9 |
| `trace.read_trades(spark, trace_dir, table, months)` | Gathers manifests and row counts on the cluster, runs `check_delivery`, returns those months as a Spark DataFrame. Refuses a local `trace_dir` inside the repo | #9 |
| `trace.clean_trades(reports)` | WRDS reports in, cleaned trades out; refuses reports from before 6 February 2012 | #10 |
| `trace.cleaning_counts(reports)` | Reports removed by each cleaning step, as totals | #10 |
| `trace.plain_trades(cleaned)` | Renames the columns the measures use (price, par volume, side, counterparty) | #10 |
| `liquidity.daily_measures(trades, snapshot_days)` | Daily Amihud and IRC per bond, trades before the snapshot day only | #11 |
| `liquidity.monthly_measures(trades, snapshot_days)` | Every measure above per bond and month, except lambda | #11 |
| `liquidity.add_lambda(measures, train_months)` | Standardises on the train months and adds lambda | #11 |
| `liquidity.join_measures(panel, measures)` | Adds the measures to the panel on bond and month | #12 |

## Acceptance tests (`tests/test_trace.py`, `tests/test_liquidity.py`)

Written before the code, on small made-up trade sets with hand-computed answers (`tests/trades.py`), and skipped until their issue is done. They run on Spark: on a local Spark in CI, and on the cluster locally when `SPARK_REMOTE` is set. When the first Spark test is unskipped (#10), CI installs Java 17 and the full `pyspark==4.2.0` for a local Spark; this machine's Java 26 is not supported by Spark 4.2, which is why local runs use the cluster.

| Test | Issue |
|---|---|
| The check accepts a complete delivery | #9 |
| The check names a missing month, a month whose row count differs from its manifest, and a missing column | #9 |
| The reader refuses a local path inside the repo | #9 |
| A cancel reported two days later removes its trade | #10 |
| A correction replaces the old version | #10 |
| A reversal removes only the trade it points to, not an identical twin | #10 |
| A dealer buy is dropped only when a matching dealer sell exists; a report with no counterparty is kept | #10 |
| When-issued, special, as-of and late-settling trades are dropped | #10 |
| Reports from before 2012 are refused | #10 |
| Each step's removals are counted | #10 |
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
- Dick-Nielsen, notes on cleaning Enhanced TRACE data (2009, 2014), the basis of the WRDS cleaned TRACE data; implemented following Tidy Finance's `clean_enhanced_trace`.
- FINRA, TRACE historic data file layout for files after 6 February 2012 (status codes).
- Feldhütter (2012), "The same bond at different prices: identifying search frictions and selling pressures", *Review of Financial Studies* 25(4), 1155-1206.
- Roll (1984), "A simple implicit measure of the effective bid-ask spread in an efficient market", *Journal of Finance* 39(4), 1127-1139.
