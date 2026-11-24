"""Made-up TRACE trades for tests, with hand-computed answers. No real data.

The measures' trades use plain names (``frame``); the cleaning's reports use WRDS columns
(``reports``). Execution times are TIME values, as the delivery stores them.
Frames are built with an explicit schema, because Spark cannot infer the type of a column
whose values are all empty.
"""

import datetime as dt
from datetime import date
from math import sqrt

DAY = date(2025, 3, 3)
SNAPSHOT = date(2025, 3, 31)  # month-end snapshot day for March 2025
SNAPSHOTS = {"2025-03": SNAPSHOT}

SCHEMA = "cusip_id string, trd_exctn_dt date, trd_seq time, msg_seq double, rptd_pr double, par_usd double"
COLUMNS = [field.split()[0] for field in SCHEMA.split(", ")]


def trade(cusip, day, seq, price, par_usd, msg_seq=None):
    """One cleaned trade, executed ``seq`` seconds after 10:00. ``msg_seq`` breaks ties (default ``seq``)."""
    return {"cusip_id": cusip, "trd_exctn_dt": day, "trd_seq": dt.time(10, 0, seq),
            "msg_seq": float(seq if msg_seq is None else msg_seq), "rptd_pr": price, "par_usd": par_usd}


def frame(spark, rows):
    """A Spark DataFrame of plain-named cleaned trades."""
    return spark.createDataFrame([tuple(row[c] for c in COLUMNS) for row in rows], schema=SCHEMA)


# Amihud: one day, three trades. Returns 101/100 - 1 = 0.01 on USD 2m, and |100.5/101 - 1| on USD 1m.
AMIHUD_DAY = [trade("B1", DAY, 1, 100.0, 1e6), trade("B1", DAY, 2, 101.0, 2e6), trade("B1", DAY, 3, 100.5, 1e6)]
AMIHUD_EXPECTED = (0.01 / 2 + (1 - 100.5 / 101) / 1) / 2

# IRC: two trades of the same size on one day at 100 and 99.
IRC_DAY = [trade("B1", DAY, 1, 100.0, 5e5), trade("B1", DAY, 2, 99.0, 5e5)]
IRC_EXPECTED = 0.01

# Roll: prices 100, 101, 100, 101, 100 on five days; changes +1, -1, +1, -1.
# Pairs (change, previous change): (-1, 1), (1, -1), (-1, 1). Sample covariance -4/3.
BOUNCE_PRICES = [100.0, 101.0, 100.0, 101.0, 100.0]
ROLL_BOUNCE = [trade("B1", date(2025, 3, 3 + i), 1, p, 1e6) for i, p in enumerate(BOUNCE_PRICES)]
ROLL_EXPECTED = 2 * sqrt(4 / 3)
# A trend with accelerating rises has positive covariance of changes (0.25): Roll is 0.
TREND_PRICES = [100.0, 101.0, 102.5, 104.5, 107.0]
ROLL_TREND = [trade("B1", date(2025, 3, 3 + i), 1, p, 1e6) for i, p in enumerate(TREND_PRICES)]

# Zero-trade share: the market trades on four days (B1 trades every day); B2 trades on two of them.
ZERO_TRADE = (
    [trade("B1", date(2025, 3, d), 1, 100.0, 1e6) for d in (3, 4, 5, 6)]
    + [trade("B2", date(2025, 3, d), 1, 100.0, 1e6) for d in (3, 5)]
)

# Snapshot day: B1's only two same-size trades are on the snapshot day, so it has no IRC.
ON_SNAPSHOT_DAY = [trade("B1", SNAPSHOT, 1, 100.0, 5e5), trade("B1", SNAPSHOT, 2, 99.0, 5e5)]

# Amihud over three days: daily values 0.01, 0.02 and 0.06. The median is 0.02; the mean would be 0.03.
AMIHUD_DAYS = [
    trade("B1", date(2025, 3, d), seq, p, 1e6)
    for d, last in [(3, 101.0), (4, 102.0), (5, 106.0)] for seq, p in [(1, 100.0), (2, last)]
]
AMIHUD_MEDIAN = 0.02

# Two trades in the same second (seq 2); the message number orders them: 100, 101, 99.
AMIHUD_TIED = [trade("B1", DAY, 1, 100.0, 1e6, msg_seq=1), trade("B1", DAY, 2, 99.0, 1e6, msg_seq=3),
               trade("B1", DAY, 2, 101.0, 1e6, msg_seq=2)]
AMIHUD_TIED_EXPECTED = (0.01 + (1 - 99 / 101)) / 2

# Four trades of one size on 3 March are not a roundtrip; a pair on 4 March is, with IRC 0.01.
IRC_FOUR_AND_PAIR = (
    [trade("B1", DAY, i, p, 5e5) for i, p in enumerate([100.0, 99.0, 98.0, 97.0], start=1)]
    + [trade("B1", date(2025, 3, 4), 1, 100.0, 5e5), trade("B1", date(2025, 3, 4), 2, 99.0, 5e5)]
)

# Three trades are two price changes: too few for Roll.
ROLL_SHORT = ROLL_BOUNCE[:3]

# A price of 0 followed by a trade, and a trade of size 0, must not stop the run: both are dropped.
ZERO_PRICE = IRC_DAY + [trade("B1", date(2025, 3, 4), 1, 0.0, 1e6), trade("B1", date(2025, 3, 4), 2, 100.0, 1e6),
                        trade("B1", date(2025, 3, 5), 1, 100.0, 0.0)]

# Cleaning works on WRDS columns. One report per row, in the order of WRDS_SCHEMA.
WRDS_SCHEMA = (
    "cusip_id string, trd_exctn_dt date, trd_exctn_tm time, trd_rpt_dt date, rptd_pr double, "
    "entrd_vol_qt double, rpt_side_cd string, cntra_mp_id string, trc_st string, msg_seq_nb double, "
    "orig_msg_seq_nb double, wis_fl string, spcl_trd_fl string, asof_cd string, days_to_sttl_ct double, "
    "stlmnt_dt date"
)
LATER = date(2025, 3, 5)  # a report date two days after the trade


def report(msg, price=100.0, trc_st="T", orig=None, rpt_day=DAY, side="S", contra="C", time="10:00:00",
           vol=1e6, wis="N", spcl=None, asof=None, sttl_days=2.0, settle=None):
    """One WRDS trade report for bond B1 executed on DAY."""
    return ("B1", DAY, dt.time.fromisoformat(time), rpt_day, price, vol, side, contra, trc_st, float(msg),
            None if orig is None else float(orig), wis, spcl, asof, sttl_days, settle)


def reports(spark, rows):
    """A Spark DataFrame of WRDS trade reports."""
    return spark.createDataFrame(rows, schema=WRDS_SCHEMA)


# A cancel (X) reported two days later repeats the message number and details of the trade it removes.
# Report 11 is an identical twin of 10 with its own message number, so it must stay.
CANCELLED_LATER = [report(10), report(11), report(10, trc_st="X", rpt_day=LATER),
                   report(12, price=101.0, time="11:00:00")]
# A correction: the old version is cancelled (C, same message number and details) and the new one (R) stays.
CORRECTED = [report(20), report(20, trc_st="C", rpt_day=LATER), report(21, price=100.5, trc_st="R", rpt_day=LATER)]
# Two identical trades; a reversal (Y) weeks later points to one of them through orig_msg_seq_nb.
REVERSED_TWIN = [report(40), report(41), report(50, trc_st="Y", orig=40, rpt_day=date(2025, 4, 1))]
# Interdealer: a dealer sell and the matching dealer buy (one trade); a dealer buy that only a customer sell
# matches, so it stays; and a report with no counterparty code, which goes.
INTERDEALER = [
    report(30, contra="D", side="S"), report(31, contra="D", side="B"),
    report(32, contra="D", side="B", price=99.0, time="12:00:00"),
    report(34, contra="C", side="S", price=99.0, time="12:30:00"),
    report(33, contra=None, side="B", time="13:00:00"),
]
# Dropped by the last step: when-issued, special, as-of, settlement 10 days out by the day count, and
# by the settlement date. Report 66 settles 3 days out and stays. A report with no report date goes earlier.
SPECIAL = [report(60), report(61, wis="Y"), report(62, spcl="S"), report(63, asof="A"), report(64, sttl_days=10.0),
           report(65, settle=date(2025, 3, 13)), report(66, settle=date(2025, 3, 6)), report(67, rpt_day=None)]
# Reported before the 2012 platform change.
OLD_PLATFORM = [report(70, rpt_day=date(2011, 6, 1))]
