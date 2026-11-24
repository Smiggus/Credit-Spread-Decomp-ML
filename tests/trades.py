"""Made-up TRACE trades for tests, with hand-computed answers. No real data.

The measures' trades use plain names (``frame``); the cleaning's reports use WRDS columns
(``reports``).
Frames are built with an explicit schema, because Spark cannot infer the type of a column
whose values are all empty.
"""

from datetime import date
from math import sqrt

DAY = date(2025, 3, 3)
SNAPSHOT = date(2025, 3, 31)  # month-end snapshot day for March 2025
SNAPSHOTS = {"2025-03": SNAPSHOT}

SCHEMA = (
    "cusip_id string, trd_exctn_dt date, trd_seq int, rptd_pr double, par_usd double, "
    "contra string, side string, status string, msg_seq string, orig_msg_seq string"
)
COLUMNS = [field.split()[0] for field in SCHEMA.split(", ")]


def trade(cusip, day, seq, price, par_usd, contra="C", side="S", status="trade", msg_seq=None, orig_msg_seq=None):
    """One trade report. ``contra`` C = customer, D = dealer; ``side`` B or S as reported."""
    return {
        "cusip_id": cusip, "trd_exctn_dt": day, "trd_seq": seq, "rptd_pr": price, "par_usd": par_usd,
        "contra": contra, "side": side, "status": status,
        "msg_seq": str(msg_seq if msg_seq is not None else seq),
        "orig_msg_seq": None if orig_msg_seq is None else str(orig_msg_seq),
    }


def frame(spark, rows):
    """A Spark DataFrame of plain-named trades."""
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

# Cleaning works on WRDS columns. One report per row, in the order of WRDS_SCHEMA.
WRDS_SCHEMA = (
    "cusip_id string, trd_exctn_dt date, trd_exctn_tm string, trd_rpt_dt date, rptd_pr double, "
    "entrd_vol_qt double, rpt_side_cd string, cntra_mp_id string, trc_st string, msg_seq_nb double, "
    "orig_msg_seq_nb double, wis_fl string, spcl_trd_fl string, asof_cd string, days_to_sttl_ct double, "
    "stlmnt_dt date"
)
LATER = date(2025, 3, 5)  # a report date two days after the trade


def report(msg, price=100.0, trc_st="T", orig=None, rpt_day=DAY, side="S", contra="C", time="10:00:00",
           vol=1e6, wis="N", spcl=None, asof=None, sttl_days=2.0, settle=None):
    """One WRDS trade report for bond B1 executed on DAY."""
    return ("B1", DAY, time, rpt_day, price, vol, side, contra, trc_st, float(msg),
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
