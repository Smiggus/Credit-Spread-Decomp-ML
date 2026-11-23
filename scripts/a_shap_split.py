"""Stage a end to end: fit on the train months, split the predicted spread on train and valid,
and write reports/a_group_shares.csv, reports/figures/a_group_shares.png and
reports/a_drop_group.csv. Totals only.

Run from the repo root: python -m scripts.a_shap_split
About 8.5 minutes on a laptop: the SHAP values for about 148,000 rows take most of it, and the
six drop-group fits about a minute.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from src.config import REPO_ROOT
from src.panel import input_groups, load_panel
from src.shap_split import drop_group_mae, fit_spread_model, group_contributions, group_shares

logger = logging.getLogger(__name__)

REPORTS = REPO_ROOT / "reports"
# Each cut is reported for the train months (in-sample) and the valid months, kept apart.
CUTS = {
    "grade": ["grade_bucket"],
    "grade_maturity": ["grade_bucket", "maturity_bucket"],
    "grade_month": ["grade_bucket", "snapshot_date"],
}
# Validated categorical palette (light surface), one colour per input group in config order.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def shares_table(contribs: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """All cuts in one long table, with a ``cut`` column naming the cut."""
    tables = [group_shares(contribs, df, ["partition"] + by).assign(cut=name) for name, by in CUTS.items()]
    table = pd.concat(tables, ignore_index=True)
    first = ["cut", "partition", "grade_bucket", "maturity_bucket", "snapshot_date", "group"]
    return table[first + ["mean_bp", "share_pct"]]


def plot_monthly_shares(table: pd.DataFrame, path) -> None:
    """Stacked monthly shares, one panel for IG and one for HY, with a line where the valid months start."""
    monthly = table[table["cut"] == "grade_month"]
    groups = list(input_groups())
    valid_start = monthly.loc[monthly["partition"] == "valid", "snapshot_date"].min()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    for ax, grade in zip(axes, ["IG", "HY"]):
        rows = monthly[monthly["grade_bucket"] == grade]
        wide = rows.pivot(index="snapshot_date", columns="group", values="share_pct")[groups]
        ax.set_facecolor(SURFACE)
        ax.stackplot(wide.index, wide.T.to_numpy(), colors=COLOURS, edgecolor=SURFACE, linewidth=1.2)
        ax.axvline(valid_start, color=INK, linestyle="--", linewidth=1)
        ax.text(valid_start, 101, "train (in-sample) ", color=MUTED, fontsize=8, va="bottom", ha="right")
        ax.text(valid_start, 101, " valid", color=MUTED, fontsize=8, va="bottom")
        ax.set_title(grade, color=INK, loc="left", fontsize=11, pad=16)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Share of mean absolute contribution (%)" if grade == "IG" else "", color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=MUTED, alpha=0.15, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    labels = [name.replace("_", " ").capitalize() for name in groups]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=5, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.suptitle(
        "Each input group's share of the mean absolute SHAP contribution, by month (base value excluded)",
        color=INK, x=0.06, ha="left",
    )
    fig.text(0.06, 0.01, "June 2025 is the gap month between train and valid and has no data.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    """Write the shares table, the chart and the drop-group check to ``reports/``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = load_panel(["train", "valid"])
    train, valid = df[df["partition"] == "train"], df[df["partition"] == "valid"]
    table = shares_table(group_contributions(fit_spread_model(train), df), df)
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "a_group_shares.csv", index=False, float_format="%.2f")
    plot_monthly_shares(table, REPORTS / "figures" / "a_group_shares.png")
    drop_group_mae(train, valid).to_csv(REPORTS / "a_drop_group.csv", index=False, float_format="%.2f")
    logger.info("Wrote the shares table, the chart and the drop-group check to %s", REPORTS)


if __name__ == "__main__":
    main()
