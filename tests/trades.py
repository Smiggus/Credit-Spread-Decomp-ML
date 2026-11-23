"""Made-up TRACE trades for tests, with hand-computed answers. No real data.

Trades use the plain status names that ``trace.clean_trades`` works on (trade, cancel,
correction, reversal); mapping the WRDS codes to these names is checked in #10.
"""

from datetime import date
from math import sqrt

DAY = date(2025, 3, 3)
SNAPSHOT = date(2025, 3, 31)  # month-end snapshot day for March 2025
SNAPSHOTS = {"2025-03": SNAPSHOT}


def trade(cusip, day, seq, price, par_usd, contra="C", side="S", status="trade", msg_seq=None, orig_msg_seq=None):
    """One trade report. ``contra`` C = customer, D = dealer; ``side`` B or S as reported."""
    return {
        "cusip_id": cusip, "trd_exctn_dt": day, "trd_seq": seq, "rptd_pr": price, "par_usd": par_usd,
        "contra": contra, "side": side, "status": status,
        "msg_seq": msg_seq if msg_seq is not None else seq, "orig_msg_seq": orig_msg_seq,
    }


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

# Cleaning.
CANCELLED = [
    trade("B1", DAY, 1, 100.0, 1e6, msg_seq=10),
    trade("B1", DAY, 2, 100.0, 1e6, status="cancel", msg_seq=11, orig_msg_seq=10),
    trade("B1", DAY, 3, 101.0, 1e6, msg_seq=12),
]
CORRECTED = [
    trade("B1", DAY, 1, 100.0, 1e6, msg_seq=20),
    trade("B1", DAY, 2, 100.5, 1e6, status="correction", msg_seq=21, orig_msg_seq=20),
]
INTERDEALER = [
    trade("B1", DAY, 1, 100.0, 1e6, contra="D", side="S", msg_seq=30),
    trade("B1", DAY, 1, 100.0, 1e6, contra="D", side="B", msg_seq=31),
]
